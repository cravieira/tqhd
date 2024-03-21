#!/usr/bin/python3

# This code is based on graphhd available in TorchHD. Some modifications were
# made to allow serializing the trained pytorch model.

import argparse
from functools import partial
from pathlib import Path
import torch
from torch import nn
from torch_geometric.datasets import TUDataset
import torchhd
import torchhd.embeddings as embeddings
from torchhd.types import VSAOptions

import common

class DatasetTuple():
    """
    This class allows compatibility with the training and test routines
    available in the "common" module.
    """
    def __init__(self, sample):
        super(DatasetTuple, self).__init__()
        self.x = sample.x
        self.edge_index = sample.edge_index

    def to(self, device):
        self.x.to(device)
        self.edge_index.to(device)

        return self

def load_dataset(dataset_dir, dataset, batch_size=1, apply_transform=False):
    # The graph datasets return a single object containing the entry and the
    # label. Create a transform that unpack the necessary values.
    def _transform(sample):
        return DatasetTuple(sample), sample.y

    transform = _transform if apply_transform else None
    #graphs = TUDataset(dataset_dir, dataset, transform=transform)
    graphs = TUDataset(dataset_dir, dataset, transform=transform)
    train_size = int(0.7 * len(graphs))
    test_size = len(graphs) - train_size

    device = common.get_device()
    generator = torch.Generator(device=device)

    train_ld, test_ld = torch.utils.data.random_split(
            graphs,
            [train_size, test_size],
            generator=generator)

    return train_ld, test_ld

class Pagerank(nn.Module):
    """
    Implement Pagerank kernel as Pytorch module to allow serialization to file.
    """
    def __init__(self):
        super().__init__()

    def sparse_stochastic_graph(self, num_nodes, edge_index):
        """
        Returns a sparse adjacency matrix of the graph G.
        The values indicate the probability of leaving a vertex.
        This means that each column sums up to one.
        """
        columns = edge_index[1] # Get the destination edges
        # Calculate the probability for each column
        values_per_column = 1.0 / torch.bincount(columns, minlength=num_nodes)
        values_per_node = values_per_column[columns]

        size = (int(num_nodes), int(num_nodes))
        return torch.sparse_coo_tensor(edge_index, values_per_node, size)

    def pagerank(self, x, edge_index):
        alpha = 0.85
        max_iter = 100
        tol = 1e-06
        # Get the number of nodes
        N = torch.tensor(x.shape[0])
        M = self.sparse_stochastic_graph(N, edge_index) * alpha
        v = torch.zeros(N, device=edge_index.device) + 1 / N
        p = torch.zeros(N, device=edge_index.device) + 1 / N
        for _ in range(max_iter):
            v_prev = v
            v = M @ v + p * (1 - alpha)

            err = (v - v_prev).abs().sum()
            # if tol != None and err < N * tol: # Original if
            if err < N * tol: # Removed tol != None check
                return v
        return v

    def forward(self, x, edge_index):
        return self.pagerank(x, edge_index)

def to_undirected(edge_index):
    """
    Returns the undirected edge_index
    [[0, 1], [1, 0]] will result in [[0], [1]]
    """
    edge_index = edge_index.sort(dim=0)[0]
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index

#def min_max_graph_size(graph_dataset):
def min_max_graph_size(args):
    """
    Find min and max number of nodes in the entire dataset based on the dataset
    selected by the argument parser.

    :param args:type: Argument parser.
    """
    def _find_min_max(loader):
        if len(loader) == 0:
            raise RuntimeError('Attempt to find max/min number of nodes in empty dataset.')

        max_num_nodes = float("-inf")
        min_num_nodes = float("inf")

        for G in loader:
            num_nodes = G.num_nodes
            max_num_nodes = max(max_num_nodes, num_nodes)
            min_num_nodes = min(min_num_nodes, num_nodes)

        return min_num_nodes, max_num_nodes

    # Get train and test loaders
    train_ld, test_ld = load_dataset(args.dataset_dir, args.dataset, apply_transform=False)
    min_train, max_train = _find_min_max(train_ld)
    min_test, max_test = _find_min_max(test_ld)

    min_nodes = min(min_train, min_test)
    max_nodes = max(max_train, max_test)

    return min_nodes, max_nodes

class GraphHD(nn.Module):
    """docstring for Encoder"""
    def __init__(
            self,
            dimensions,
            num_classes,
            size,
            *,
            vsa: VSAOptions='MAP',
            dtype_enc=torch.get_default_dtype(),
            dtype_am=torch.get_default_dtype(),
            device='cpu',
            **kwargs
        ):
        super(GraphHD, self).__init__()
        self.dimensions = dimensions
        self.dtype_enc = dtype_enc
        self.dtype_am = dtype_am
        self.vsa: VSAOptions = vsa
        self.device = device

        self.ids = embeddings.Random(
                size,
                dimensions,
                vsa=self.vsa,
                dtype=self.dtype_enc,
                device=device)

        self.am = common.pick_am_model(vsa, dimensions, num_classes, **kwargs)

    def create_am(self):
        self.am.train_am()

    def encode(self, sample, pr):
        x = sample.x
        edge_index = sample.edge_index

        pr_sort, pr_argsort = pr.sort()

        # Using x.shape[0] instead of num_nodes to allow Pytorch tracing
        #num_nodes = torch.tensor(x.shape[0])

        node_id_hvs = torch.zeros(
                (x.shape[0], self.dimensions),
                device=self.device,
                dtype=self.dtype_enc
            )
        node_id_hvs[pr_argsort] = self.ids.weight[: x.shape[0]]

        #row, col = self.to_undirected(edge_index)
        G = to_undirected(edge_index)

        hvs = torchhd.bind(node_id_hvs[G[0]], node_id_hvs[G[1]])
        enc = torchhd.multiset(hvs)

        # Return hypervectors in the shape [1, dimension]
        return enc.unsqueeze_(0)

    #def forward(self, x, edge_index):
    def forward(self, sample, pr):
        enc = self.encode(sample, pr)
        return self.am.search(enc)

class ModelPipeline(nn.Module):
    """docstring for App"""
    def __init__(self, preprocessing, hdc):
        super().__init__()
        self.preprocessing = preprocessing
        self.hdc = hdc
        self.am = hdc.am

    def create_am(self):
        self.hdc.create_am()

    def encode(self, sample):
        x = sample.x
        edge_index = sample.edge_index

        pr = self.preprocessing(x, edge_index)
        return self.hdc.encode(sample, pr)

    def forward(self, sample):
        x = sample.x
        edge_index = sample.edge_index

        pr = self.preprocessing(x, edge_index)
        return self.hdc(sample, pr)

    def to(self, device):
        self.preprocessing.to(device)
        self.hdc.to(device)
        return self

def main():
    # Enable parser #
    parser = argparse.ArgumentParser(description='Train the GraphHD HDC model.')
    common.add_default_arguments(parser)
    common.add_default_hdc_arguments(parser)
    common.add_am_arguments(parser)

    datasets = {
            'DD',
            'ENZYMES',
            'MUTAG',
            'NCI1',
            'PROTEINS',
            'PTC_FM'
            }
    default_dataset = 'MUTAG'
    parser.add_argument('--dataset',
            choices=datasets,
            default='MUTAG',
            help=f'Select the used TUDataset among the available options. Defaults to {default_dataset}.'
            )
    args = parser.parse_args()
    am_args = common.parse_am_group(parser, args)

    vsa = args.vsa
    if vsa != "BSC":
        dtype_enc = common.map_dtype(args.dtype_enc)
        dtype_am = common.map_dtype(args.dtype_am)
        model_name = f'{vsa.lower()}-enc{args.dtype_enc}-am{args.dtype_am}-d{args.vector_size}.pt'
    else:
        dtype_enc = torch.bool
        dtype_am = torch.bool
        model_name = f'{vsa.lower()}-d{args.vector_size}.pt'

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir+'/'+model_name

    if args.redirect_stdout:
        log_path = model_path.removesuffix('.pt')+'.log'
        common.redirect_stdout(log_path)

    dataset = args.dataset
    train_ld, test_ld = load_dataset(args.dataset_dir, dataset, apply_transform=True)

    common.set_random_seed(args.seed)
    device = common.set_device(args.device)

    # Search for the max number of nodes in all graphs available in the dataset
    _, max_graph_size = min_max_graph_size(args)
    num_classes = int(train_ld.dataset.y.max()+1)

    constructor = partial(
            GraphHD,
            args.vector_size,
            num_classes,
            max_graph_size,
            vsa=vsa,
            dtype_enc=dtype_enc,
            dtype_am=dtype_am,
            device=device,
            **am_args)
    hdc_model = common.args_pick_model(args, constructor)
    pagerank = Pagerank()
    model = ModelPipeline(pagerank, hdc_model)

    model = common.args_train_hdc(args, model, train_ld, test_ld=test_ld)
    accuracy = common.args_test_hdc(args, model, test_ld, num_classes)

    common.save_results(args, hdc_model, train_ld, accuracy)

if __name__ == '__main__':
    main()

