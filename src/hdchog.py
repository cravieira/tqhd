#!/usr/bin/python3

import argparse
from typing import Callable, Optional, Tuple
from functools import partial
from pathlib import Path
import torch
from torch.nn import Module
import torchhd
import torchhd.embeddings as embeddings
from torchhd.types import VSAOptions
from skimage import feature
import re
import math
from torch import nn
import torch.nn.functional as F

import common

from torchvision.datasets import MNIST, FashionMNIST
import torchvision

def sk_hog(
        img: torch.Tensor,
        pixels_per_cell: Tuple[int, int] = (3,3),
        bins: int = 9
    ):
    """
    Skimage HOG preprocessing.

    :param img: The image to be processed.
    :param pixels_per_cell: Number of pixels in each cell.
    :param bins: Number of histogram bins.
    """
    # Normalize to the range [0, 1]
    img_scaled = img / 255.0

    (hog_desc, hog_image) = feature.hog(
        img_scaled,
        orientations=bins,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=(1, 1),
        transform_sqrt=True,
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=False)
    t = torch.Tensor(hog_desc)

    # Make sure the returned tensor is suitted for HDCHOG as it requires a 2D
    # tensor in the shape [cells, bins].
    hog_tensor = torch.unsqueeze(t, dim=0)
    flat_cells = torch.flatten(hog_tensor, start_dim=0, end_dim=-2)

    return flat_cells

class HOGLayer(nn.Module):
    """
    A Pytorch implementation of the Histogram of Oriented Gradients feature
    extractor. This implementation is based on [1] with slight modifications.

    [1]: https://gist.github.com/etienne87/b79c6b4aa0ceb2cff554c32a7079fa5a
    """
    # Original
    #def __init__(self, nbins=10, pool=4, max_angle=math.pi, stride=1, padding=1, dilation=1):
    def __init__(self, nbins=11, pool=4, max_angle=math.pi, stride=1, padding=0, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool = pool
        self.max_angle = max_angle
        #mat = torch.FloatTensor([[1, 0, -1],
        #                         [2, 0, -2],
        #                         [1, 0, -1]])
        mat = torch.FloatTensor([[0, 0, 0],
                                 [1, 0, -1],
                                 [0, 0, 0]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:,None,:,:])
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    def forward(self, x):
        with torch.no_grad():
            gxy = F.conv2d(x, self.weight, None, self.stride,
                            self.padding, self.dilation, 1)
            #2. Mag/ Phase
            mag = gxy.norm(dim=1)
            norm = mag[:,None,:,:]
            phase = torch.atan2(gxy[:,0,:,:], gxy[:,1,:,:])

            #3. Binning Mag with linear interpolation
            phase_int = phase / self.max_angle * self.nbins
            phase_int = phase_int[:,None,:,:]

            n, c, h, w = gxy.shape
            out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
            out.scatter_(1, phase_int.floor().long()%self.nbins, norm)
            out.scatter_add_(1, phase_int.ceil().long()%self.nbins, 1 - norm)

            return self.pooler(out)

def dataset_transformation(x, transform: Callable):
    """
    Apply a HOG transformation to a dataset sample and transform it to a shape
    accepted by the HDC pipeline.

    :param x: An image object.
    :param transform: The HOG transformation to apply.
    """
    trans = torchvision.transforms.ToTensor()
    tensor = trans(x).squeeze()
    hog = transform(tensor)

    flat_cells = torch.flatten(hog, start_dim=0, end_dim=-2)
    hog = flat_cells

    return hog

def load_dataset(
        dataset: str = 'MNIST',
        dataset_dir: str = '_data',
        batch_size: int = 1,
        transform: Optional[Callable] = None,
        num_workers: int = 0):
    """
    Download the required dataset and return train and test loaders.

    :param dataset: Name of the dataset. Only "MNIST" and "FashionMNIST" are
    accepted.
    :param dataset_dir: Path to dataset directory where the downloaded datasets
    should be downloaded to.
    :param batch_size: Batch size used by data loaders.
    :param transform: A transformation to be applied to the loaded data.
    :param num_workers: Number of data loader workers.
    :raises RuntimeError: Raises exception if the dataset argument is not
    recognized.
    """

    if dataset == 'MNIST':
        train_ld, test_ld = common.load_dataset(dataset_dir, MNIST, batch_size=batch_size, transform=transform, num_workers=num_workers)
    elif dataset == 'FashionMNIST':
        train_ld, test_ld = common.load_dataset(dataset_dir, FashionMNIST, batch_size=batch_size, transform=transform, num_workers=num_workers)
    else:
        raise RuntimeError(f'Invalid dataset name: {dataset}')

    return train_ld, test_ld

class HDCHOG(Module):
    """
    An HDC model for image classification that works on the output of the
    Histogram of Oriented Gradients (HOG). The input must be a 2D tensor in the
    shape [cells, bins].

    The encoder used in this class is based on the paper "HDCOG: A Lightweight
    Hyperdimensional Computing Framework with Feature Extraction".
    """
    def __init__(
            self,
            dimensions,
            cells,
            levels,
            num_classes,
            bins=9,
            *,
            vsa: VSAOptions = 'MAP',
            dtype_enc=torch.get_default_dtype(),
            dtype_am=torch.get_default_dtype(),
            **kwargs
        ):
        super().__init__()
        self.dimensions = dimensions
        self.dtype_enc = dtype_enc
        self.dtype_am = dtype_am
        self.vsa: VSAOptions = vsa

        # Each cell is represented by a unique ID vector
        self.cell_hv = embeddings.Random(cells, dimensions, vsa=self.vsa, dtype=self.dtype_enc, **kwargs)
        # Each bin in a histogram is represented by an orientation vector
        self.ori_hv = embeddings.Random(bins, dimensions, vsa=self.vsa, dtype=self.dtype_enc, **kwargs)
        # Embedding for histogram magnitude values
        self.mag_hv = embeddings.Thermometer(levels, dimensions, low=0., high=1.0, vsa=self.vsa, dtype=self.dtype_enc, **kwargs)

        self.am = common.pick_am_model(vsa, dimensions, num_classes, **kwargs)

        # WIP: Support to MCR
        self.vsa_kwargs = {}
        if vsa == 'MCR':
            self.vsa_kwargs['mod'] = kwargs['mod']

    def create_am(self):
        self.am.train_am()

    def encode(self, x):
        mags = self.mag_hv(x)
        bins = torchhd.bind(self.ori_hv.weight, mags, **self.vsa_kwargs)
        grads = torchhd.multibundle(bins, **self.vsa_kwargs)
        cells = torchhd.bind(self.cell_hv.weight, grads, **self.vsa_kwargs)
        mat_hv = torchhd.multibundle(cells, **self.vsa_kwargs)
        return mat_hv

    def search(self, x):
        return self.am.search(x)

    def forward(self, x):
        enc = self.encode(x)
        return self.search(enc)


def main():
    # Enable parser #
    parser = argparse.ArgumentParser(description='Train HDFace HDC model.')
    datasets = ['MNIST', 'FashionMNIST']
    default_dataset = 'MNIST'
    parser.add_argument(
            '--dataset',
            type=str,
            default=default_dataset,
            choices=datasets,
            help=f'Selects the dataset used. Default: "{default_dataset}".'
            )
    default_dataloader_workers = 0
    parser.add_argument(
            '--dataloader-workers',
            type=int,
            default=default_dataloader_workers,
            help=f'Selects the number of workers used in dataset loading. Must'
            f'be an integer >= 0. Default: {default_dataloader_workers}.'
            )

    default_hog_algorithm = "torch"
    parser.add_argument(
            '--hog-algorithm',
            type=str,
            default=default_hog_algorithm,
            choices=['torch', 'skimage'],
            help=f'Chooses the HOG preprocessing implementation. The '
            'implementation might affect the accuracy of the model and the '
            'training time. The torch implementation trains several times faster'
            ' than the skimage, but achieves less accuracy. Default:'
            f'"{default_hog_algorithm}".'
            )

    def _transform_hog_arg(arg: str) -> Tuple[int, int]:
        if not re.fullmatch(r'\d+,\d+', arg):
            raise RuntimeError(f'Failed to parse --hog-cell parameter. Invalid argument "{arg}".')

        splitted = arg.split(sep=',')
        numbers = [n.strip() for n in splitted]
        return (int(numbers[0]), int(numbers[1]))

    default_hog_cell = '4,4'
    parser.add_argument(
            '--hog-cell',
            type=_transform_hog_arg,
            default=default_hog_cell,
            help=f'Select the cell size used in HOG preprocessing. The '
            'parameter must be two numbers (X- and Y-sizes) separated by a '
            f'comma. Default: "{default_hog_cell}".'
            )

    default_hog_bins = 9
    parser.add_argument(
            '--hog-bins',
            type=int,
            default=default_hog_bins,
            help=f'Select the number of histogram bins used in HoG. Default: {default_hog_bins}.'
            )

    common.add_default_arguments(parser)
    common.add_default_hdc_arguments(parser)
    common.add_am_arguments(parser)

    args = parser.parse_args()
    am_args = common.parse_am_group(parser, args)

    vsa_args = common.args_parse_vsa(args)
    #if vsa != "BSC":
    #    dtype_enc = common.map_dtype(args.dtype_enc)
    #    dtype_am = common.map_dtype(args.dtype_am)
    #    model_name = f'{vsa.lower()}-enc{args.dtype_enc}-am{args.dtype_am}-d{args.vector_size}.pt'
    #else:
    #    dtype_enc = torch.bool
    #    dtype_am = torch.bool
    #    model_name = f'{vsa.lower()}-d{args.vector_size}.pt'

    #if vsa == 'FHRR':
    #    dtype_enc = torch.complex64
    #    dtype_am = torch.complex64

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    #model_path = args.model_dir+'/'+model_name

    #if args.redirect_stdout:
    #    log_path = model_path.removesuffix('.pt')+'.log'
    #    common.redirect_stdout(log_path)

    common.set_random_seed(args.seed)
    device = common.set_device(args.device)
    # Pytorch batch size
    BATCH_SIZE = 1
    # Quantization levels
    LEVELS = 11

    # The commands below are necessary if multiple dataloader jobs are used
    # with CUDA device.
    #import torch.multiprocessing as mp
    #mp.set_start_method('spawn')

    transform_backend = None
    if args.hog_algorithm == 'torch':
        # Transform the dataset sample from [width, height] to [batch, color,
        # width, height] since it is the input shape accepted by HOGLayer.
        transform_sample = lambda x: x.view(1, 1, *x.shape)
        hog_layer = HOGLayer(
                nbins=args.hog_bins,
                pool=args.hog_cell[0],
                )
        # Reorder the tensor shape produced by HOGLayer to a shape accepted by the HDCHOG
        transform_hog = lambda hog: torch.permute(hog, (0, 2, 3, 1))

        transform_backend = lambda x: \
            transform_hog(
                hog_layer(
                    transform_sample(x)
                )
            )
    elif args.hog_algorithm == 'skimage':
        transform_backend = lambda x: sk_hog(
                x,
                pixels_per_cell=args.hog_cell,
                bins=args.hog_bins,
                )
    else:
        raise RuntimeError(f'Invalid HOG algorithm: "{args.hog_algorithm}".')

    transform = lambda x: dataset_transformation(
            x,
            transform_backend
            )
    train_ld, test_ld = load_dataset(
            args.dataset,
            args.dataset_dir,
            batch_size=BATCH_SIZE,
            transform=transform,
            num_workers=args.dataloader_workers)

    num_classes = len(train_ld.dataset.classes)
    dataset_sample = train_ld.dataset[0]
    # Get the number of cells produced by the HOG
    no_cells = dataset_sample[0].shape[0]

    constructor = partial(
            HDCHOG,
            args.vector_size,
            no_cells,
            LEVELS,
            num_classes,
            bins=args.hog_bins,
            #vsa=vsa,
            #dtype_enc=dtype_enc,
            #dtype_am=dtype_am,
            **vsa_args,
            **am_args,
            )
    model = common.args_pick_model(args, constructor)

    model = common.args_train_hdc(args, model, train_ld, test_ld=test_ld)
    accuracy = common.args_test_hdc(args, model, test_ld, num_classes)

    common.save_results(args, model, train_ld, accuracy)

if __name__ == '__main__':
    main()


