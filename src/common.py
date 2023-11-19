import argparse
from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Callable, Optional
from am.am import AMBsc, AMMap, AMSignQuantize, AMThermometerQuantile, AMThermometerDeviation, PQHDC
import torch
import torchmetrics
import torchhd
from torchhd.functional import ensure_vsa_tensor, multibundle
from tqdm import tqdm

from am.learning import Centroid, OnlineHD
from am.prediction import Fault, Normal

_current_pytorch_device = "cpu"

def create_path(path: Path):
    """
    Create specified directory hierarchy.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def add_default_arguments(parser: ArgumentParser):
    """Add default parser arguments used in all scripts"""
    default_dataset_dir = './_data'
    default_model_dir = './_model'
    default_seed = 0
    default_device = 'cpu'

    device_choices = ['cpu', 'cuda']

    parser.add_argument(
            '-d', '--dataset-dir',
            default=default_dataset_dir,
            help=f'Dataset prefix path. This path contains the downloaded\
            datasets. Notice that a new folder will be created inside the given\
            path to store the dataset. The prefix path is created if necessary.\
            Defaults to \"{default_dataset_dir}\".')
    parser.add_argument(
            '-m', '--model-dir',
            default=default_model_dir,
            help=f'Model output directory. This path is used as prefix for the\
            generated models. The directory is created if necessary. Defaults\
            to \"{default_model_dir}\".')
    parser.add_argument('-a', '--accuracy-file',
            default=None,
            help=f'Print accuracy of the model to the specified file.'
            )
    parser.add_argument('-r', '--redirect-stdout',
            action='store_true',
            help=f'Redirect stdout to file. Defaults to the same path as the model created.'
            )
    parser.add_argument('--seed',
            default=default_seed,
            type=int,
            help=f'Choose the random seed used. Defaults to \"{default_seed}\".'
            )
    parser.add_argument('--device',
            default=default_device,
            type=str,
            choices=device_choices,
            help=f'Choose the device used in experiments. Defaults to \
            \"{default_device}\".'
            )
    parser.add_argument('--export',
            default=None,
            help=f'Export model to the specified path. The parent folders are \
            created if necessary.'
            )
    parser.add_argument('--save-model',
            default=None,
            help='Serialize the trained model to the given path. Parent folders'
            ' are created if necessary.'
            )
    parser.add_argument('--load-model',
            default=None,
            help='Load the trained model from the given path.'
            )
    parser.add_argument('--skip-train',
            action='store_true',
            default=False,
            help='Skip train of a model.'
            )
    parser.add_argument('--skip-test',
            action='store_true',
            default=False,
            help='Skip test of a model.'
            )

    return parser

_map_dtype = {
        'i16': torch.int16,
        'i32': torch.int32,
        'i64': torch.int64,
        'f32': torch.float32,
        'f64': torch.float64,
        'bool': torch.bool,
}

def map_dtype(key: str):
    """docstring for map_dtype"""
    return _map_dtype[key]

def add_default_hdc_arguments(parser: ArgumentParser):
    # Default HDC model tuning parameters
    vsa_type = { 'MAP', 'BSC' }
    default_vsa = 'MAP'
    default_dtype_enc = 'f32'
    default_dtype_am = 'f32'
    default_dim = 1000

    # Model tuning parameters
    parser.add_argument('--vsa',
            choices=vsa_type,
            default='MAP',
            help=f'Select the used VSA model. Defaults to \"{default_vsa}\".'
            )
    parser.add_argument('--dtype-enc',
            choices=_map_dtype.keys(),
            default=default_dtype_enc,
            help=f'Select dtype used in encode operations. If the VSA model is binary, then this parameter is ignored and bool is used. Defaults to \"{default_dtype_enc}\".'
            )
    parser.add_argument('--dtype-am',
            choices=_map_dtype.keys(),
            default=default_dtype_am,
            help=f'Select dtype used in the Associative Memory search. If the VSA model binary, then this parameter is ignored and bool is used. Defaults to \"{default_dtype_enc}\".'
            )
    parser.add_argument('--vector-size',
            type=int,
            default=default_dim,
            help=f'Select the number of dimensions used in the vectors. Defaults to \"{default_dim}\".'
            )

    return parser

# Based on https://stackoverflow.com/a/64259328
def _float_range(mini, maxi):
    """Return function handle of an argument type function for
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError(f"must be in range [ {mini} .. {maxi} ]")
        return f

    # Return function handle to checking function
    return float_range_checker

def add_am_arguments(parser: ArgumentParser):
    default_am = 'V'
    am_types = ['V', 'Q', 'B', 'TQ', 'TD', 'SQ', 'PQHDC']

    group = parser.add_argument_group('Associative Memory', 'Allows for fine control of AM parameters.')

    group.add_argument(
            '--am-type',
            choices=am_types,
            default=default_am,
            help='Chooses the AM type used. V is the regular AM, Q is a quantized MAP, and B is a MAP to BSC transformed vector.'
            )
    default_intervals = 4
    group.add_argument(
            '--am-intervals',
            default=default_intervals,
            type=int,
            help=f'Choose the number of intervals used with Q, B, and T AMs. This argument is ignored with V AM. Defaults to {default_intervals}'
            )

    default_bits = 8
    group.add_argument(
            '--am-bits',
            default=default_bits,
            type=int,
            help=f'Choose the number of bits expansion used in AM B and T. Defaults to {default_bits}.'
            )

    default_quantile = 0.25
    group.add_argument(
            '--am-tq-quantile',
            default=default_quantile,
            type=_float_range(0.0, 0.5),
            help=f'Choose the default quantile evaluated when assigning the quantization poles used in AMTQuantile. Only float values between [0, 0.5] are accepted. Defaults to {default_quantile}.'
            )

    default_deviation = 1.0
    group.add_argument(
            '--am-td-deviation',
            default=default_deviation,
            type=_float_range(0.0, float('inf')),
            help=f'Choose the multiplier used with the standard deviation when assigning the quantization poles of AMTDeviation. Only positve float values are accepted. Defaults to {default_deviation}.'
            )

    learning_types = ['Centroid', 'OnlineHD']
    default_learning = 'Centroid'
    group.add_argument(
            '--am-learning',
            default=default_learning,
            choices=learning_types,
            type=str,
            help=f'Chooses the learning strategy used by the AM. Defaults to {default_learning}.'
            )

    # Learning rate used in OnlineHD
    default_onlinehd_lr = 1.0
    group.add_argument(
            '--am-onlinehd-lr',
            default=default_onlinehd_lr,
            type=float,
            help=f'Chooses the learning rate used by the OnlineHD strategy. Defaults to {default_onlinehd_lr}.'
            )

    prediction_types = ['Normal', 'Fault']
    default_prediction = 'Normal'
    group.add_argument(
            '--am-prediction',
            default=default_prediction,
            choices=prediction_types,
            type=str,
            help=f'Chooses the prediction strategy used by the AM. Defaults to {default_prediction}.'
            )

    # Fault rate used with faulty prediction
    default_fr = 0.1
    group.add_argument(
            '--am-fault-rate',
            default=default_fr,
            type=float,
            help=f'Chooses the fault rate used by the Fault AM strategy. Defaults to {default_fr}.'
            )

    # Fault rate used with faulty prediction
    default_projections = 2
    group.add_argument(
            '--am-pqhdc-projections',
            default=default_projections,
            type=int,
            help=f'Chooses the number of projections used by PQHDC. Defaults to {default_projections}.'
            )

def parse_am_group(parser, args):
    arg_groups={}
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=argparse.Namespace(**group_dict)

    am_group = arg_groups['Associative Memory']
    return am_group.__dict__

def args_pick_learning(args):
    """
    Get the learning strategy based on the argument parser.
    """
    learning = None
    if args.am_learning == 'Centroid':
        learning = Centroid()
    elif args.am_learning == 'OnlineHD':
        lr = args.am_onlinehd_lr
        learning = OnlineHD(lr)
    else:
        RuntimeError(f"Invalid learning strategy \"{args.am_learning}\"")

    return learning

def args_pick_prediction(args):
    """
    Get the prediction strategy based on the argument parser.
    """
    prediction = Normal()
    if args.am_prediction == 'Fault':
        prediction = Fault(args.am_fault_rate)

    return prediction

def pick_am_model(
        vsa,
        dim,
        num_classes,
        am_type='V',
        am_learning='Centroid',
        am_prediction='Normal',
        **kwargs):
    learning = None
    prediction = None

    if am_learning == 'Centroid':
        learning = Centroid()
    elif am_learning == 'OnlineHD':
        lr = kwargs['am_onlinehd_lr']
        learning = OnlineHD(lr)
    else:
        RuntimeError(f"Invalid learning strategy \"{am_learning}\"")

    if am_prediction == 'Normal':
        prediction = Normal()
    elif am_prediction == 'Fault':
        prediction = Fault(kwargs['am_fault_rate'])

    if am_type == 'V':
        if vsa == 'MAP':
            am = AMMap(dim, num_classes, learning=learning, **kwargs)
        else:
            am = AMBsc(dim, num_classes, learning=learning, prediction=prediction, **kwargs)
    elif am_type == 'TQ':
        bits = kwargs['am_bits']
        intervals = kwargs['am_intervals']
        quantile = kwargs['am_tq_quantile']
        am = AMThermometerQuantile(dim, num_classes, bits, intervals, quantile, learning=learning, prediction=prediction, **kwargs)
    elif am_type == 'TD':
        bits = kwargs['am_bits']
        intervals = kwargs['am_intervals']
        deviation = kwargs['am_td_deviation']
        am = AMThermometerDeviation(dim, num_classes, bits, intervals, deviation, learning=learning, prediction=prediction, **kwargs)
    elif am_type == 'SQ':
        am = AMSignQuantize(dim, num_classes, learning=learning, prediction=prediction, **kwargs)
    elif am_type == 'PQHDC':
        projections = kwargs['am_pqhdc_projections']
        am = PQHDC(dim, num_classes, projections=projections, learning=learning, prediction=prediction, **kwargs)

    else:
        raise RuntimeError(f'Unrecognized AM type: {am_type}.')

    return am

def redirect_stdout(path):
    """Redirect stdout to a file"""
    sys.stdout = open(path, 'w')

def dump_accuracy(path, accuracy):
    """Dump accuracy to file"""
    create_path(Path(path).parent)
    with open(path, 'w') as f:
            print(accuracy, file=f)

def set_random_seed(seed=0):
    """Set random seed to favor reproducible experiments"""
    # Set random seed
    # These seed setings were did not change the accuracy, but the pytorch
    # documentation recommends setting them:
    # https://pytorch.org/docs/stable/notes/randomness.html
    #import random
    #random.seed(0)
    #import numpy as np
    #np.random.seed(0)
    # The following seed setting changed the results.
    torch.manual_seed(seed)
    torch.set_num_threads(1)

def set_device(device='cpu'):
    """Set device used in experiments."""
    torch.set_default_device(device)
    print(f'Using {device} device')
    global _current_pytorch_device
    _current_pytorch_device = device
    return device

def get_device():
    """Get pytorch default device currently set."""
    global _current_pytorch_device
    return _current_pytorch_device

def load_dataset(dataset_dir, F, batch_size=1, transform=None):
    device = get_device()
    generator = torch.Generator(device=device)

    train_ds = F(dataset_dir, train=True, transform=transform, download=True)
    train_ld = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            generator=generator,
            shuffle=False)

    test_ds = F(dataset_dir, train=False, transform=transform, download=True)
    test_ld = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            generator=generator,
            shuffle=True)

    return train_ld, test_ld

def export_model(model, model_path, dataset_loader):
    '''
    Export the model to file. This function changes the torch default device
    to CPU while exporting the model. The model parameter is also moved to the
    CPU, but it is the caller's responsability to move it back.
    '''
    create_path(Path(model_path).parent)
    # Get a single input from the dataset in the shape [1, DIM]
    device = get_device()
    set_device("cpu")
    dummy_input = dataset_loader.dataset[0][0].unsqueeze(0)

    # Export torchscript
    model = model.to("cpu")
    dummy_input = dummy_input.to("cpu")
    traced_script_module = torch.jit.trace(model, dummy_input)
    print("Code:")
    print(traced_script_module.code)
    print("Graph:")
    print(traced_script_module.graph)
    traced_script_module.save(model_path)
    set_device(device)

def save_model(model, path: str | Path):
    create_path(Path(path).parent)
    torch.save(model, path)

def load_model(path: str | Path):
    model = torch.load(path)
    if model is None:
        raise RuntimeError(f'Failed to load model at {str(path)}')
    return model

def args_pick_model(args, model_constructor: Callable):
    '''
    Choose between the model that can be instantiated by the constructor and
    the model that can be loaded given to the ArgumentParser.
    '''
    model = None
    if args.load_model:
        model = load_model(args.load_model)
    else:
        model = model_constructor()

    model.to(args.device)

    return model

def save_results(
        args,
        model: torch.nn.Module,
        train_ld,
        accuracy: Optional[float],
        ):
    '''
    Save the results of a training/testing session according to the options
    given to the argument parser.
    '''
    if args.export:
        export_model(model, args.export, train_ld)
    if args.save_model:
        save_model(model, args.save_model)

    # Print accuracy to file
    if args.accuracy_file:
        if type(accuracy) is not float:
            raise RuntimeError(
                'Attempt to create accuracy file without accuracy value.'
                )
        dump_accuracy(args.accuracy_file, accuracy)

def train_hdc(model, train_ld, device):
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc='Training'):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = model.encode(samples)
            model.am.update(samples_hv, labels)

    model.create_am()
    return model

def test_hdc(model, test_ld, num_classes, device):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    model.to(device)

    with torch.no_grad():
        for samples, labels in tqdm(test_ld, desc='Testing'):
            samples = samples.to(device)

            outputs = model(samples)
            outputs = outputs.type(torch.float)
            accuracy.update(outputs, labels)

    final_acc = accuracy.compute().item() * 100
    print(f'Testing accuracy of {final_acc:.3f}%')
    return final_acc

def args_train_hdc(args, model, train_ld):
    '''
    High level function to control training according to the parameters given
    to the ArgumentParser.
    '''
    if args.skip_train:
        return model
    return train_hdc(model, train_ld, args.device)

def args_test_hdc(args, model, test_ld, num_classes) -> Optional[float]:
    '''
    High level function to control test according to the parameters given to
    the ArgumentParser.
    '''
    if args.skip_test:
        return None
    return test_hdc(model, test_ld, num_classes, device=args.device)

# Math and transformations #
def normalize(t, dtype=torch.get_default_dtype()):
    # Normalize input
    dot = torch.sum(t*t, dim=-1, dtype=dtype)
    mag = torch.sqrt(dot)
    # Adjust shape
    mag = mag.unsqueeze(1)
    #norm = t/mag
    norm = torch.div(t, mag)

    return norm

def _create_poles(input: torch.Tensor, intervals, quantile=0.25):
    """
    Create quantization poles used by the AM object in its transformation
    based on the input. This function assumes that the input is normalized.
    """
    # Create 1D tensor with quantization poles
    # Start linspace at the furtherst quantile
    qth = quantile
    q = torch.tensor([qth, 1-qth])
    q_vals = input.quantile(q)
    # Pick the value furtherst from 0
    start = torch.max(torch.abs(q_vals))
    poles = torch.linspace(-start.item(), start.item(), steps=intervals)
    return poles

def _find_poles(input: torch.Tensor, poles: torch.Tensor):
    """
    Find the closest quantization pole to the given input. Returns a tensor
    of indices to the closest poles.
    """
    # Make poles a tensor of [Intervals, 1, 1 ..., 1] depending on the
    # shape of the input.
    view = [1] *len(input.shape)
    poles = poles.view(-1, *view)
    # Get absolute difference to each pole
    diffs = torch.abs(input - poles)
    _, inds = torch.min(diffs, dim=0)
    return inds

def _map_to_binary_encoding(input: torch.Tensor, inds, table: torch.Tensor):
    """
    Map the given input to the encoding table according to inds.
    """
    input_dim = input.shape[-1]
    table_entry_dim = table.shape[-1]
    res = table[inds].view((-1, input_dim*table_entry_dim))
    return res

# Associative Memories #
class AssociativeMemory():
    """
    Implements an Associative Memory for different VSA models and data
    types.
    """
    def __init__(self, vsa, dtype):
        super(AssociativeMemory, self).__init__()
        self.vsa = vsa
        self.dtype = dtype

    def create_am(self, tensors):
        # Achieves better accuracy with hard_quantize in encode. Only works
        # with float
        if self.dtype == torch.float32:
            norms = tensors.norm(dim=1, keepdim=True)
            eps=1e-12
            norms.clamp_(min=eps)
            norm = tensors.div(norms)
        else:
            # Provides worse accuracy
            norm = tensors

        self.am = ensure_vsa_tensor(norm, vsa=self.vsa, dtype=self.dtype)

    def search(self, query):
        """Search for the most similar entry in the AM for the given input."""
        if self.vsa == 'MAP':
            #logit = torchhd.dot_similarity(self.am, enc, dtype=self.dtype_am)
            logit = torchhd.cosine_similarity(self.am, query)
        else:
            logit = torchhd.hamming_similarity(self.am, query)
        return logit

class AssociativeMemoryB():
    """
    Implements an Associative Memory that transforms MAP to BSC.
    """
    def __init__(self, intervals: int, bits: int, quantile=0.25):
        super().__init__()
        self.intervals = intervals
        self.bits = bits
        self.quantile = quantile
        self.possible_encodings = bits//2 + 1
        self.vsa = 'BSC'

        if intervals >= self.possible_encodings:
            raise RuntimeError(f'Number of intervals ({intervals}) is to big for the number of bits given ({bits}). Only {self.possible_encodings} are possible.')

        # Create encodings
        self.enc_table = self.encode_table(self.bits, self.intervals)
        self.poles = None

    def circulant(self, tensor, dim):
        # Based on: https://stackoverflow.com/questions/69820726/is-there-a-way-to-compute-a-circulant-matrix-in-pytorch
        """get a circulant version of the tensor along the {dim} dimension.

        The additional axis is appended as the last dimension.
        E.g. tensor=[0,1,2], dim=0 --> [[0,1,2],[2,0,1],[1,2,0]]"""
        S = tensor.shape[dim]
        flipped = tensor.flip((dim,))
        tmp = torch.cat([flipped, torch.narrow(flipped, dim=dim, start=0, length=S-1)], dim=dim)
        mat = tmp.unfold(dim, S, 1).flip((-1,))
        return mat.squeeze_(0)

    def encode_table(self, bits, entries):
        # Create [1, 1, ..., 0, 0] pattern with equal numbers of 0's and 1's
        pattern = torch.zeros((1, bits), dtype=torch.int)
        pattern[..., :bits//2] = 1

        # Create circular matrix
        circular_mat = self.circulant(pattern, 1)

        # Create bit wave pattern from circular matrix
        possible_encodings = bits//2 + 1
        wave_table = circular_mat[..., 0:possible_encodings, :]

        # Remove middle entries
        rows = entries//2 # Number of top and below rows used
        top_rows = wave_table[..., 0:rows, :]
        bot_rows = wave_table[..., -rows:, :]

        # Create the final table without the rows in the middle
        enc_table = torch.vstack([top_rows, bot_rows])
        return enc_table

    def _get_poles(self, input: torch.Tensor):
        """
        Return the poles tensor used by this AM. Creates the quantization poles
        values if they do not exist yet based on the given input. Assumes the
        input is normalized.
        """
        if self.poles is None:
            self.poles = _create_poles(input, self.intervals, self.quantile)
        return self.poles

        # Alternative approach generating the quantizaiton poles to each input
        #return self._create_poles(input, self.quantile)

    def transform(self, input):
        t = torchhd.functional.ensure_vsa_tensor(input, vsa='MAP', dtype=torch.float)
        dtype = torch.float

        norm = normalize(t, dtype=dtype)

        poles = self._get_poles(norm)
        inds = _find_poles(norm, poles)

        # Create plain BSC vectors
        res = _map_to_binary_encoding(norm, inds, self.enc_table)
        return res

    def create_am(self, tensors):
        self.am = self.transform(tensors)

    def search(self, query):
        vector = self.transform(query)
        logit = torchhd.hamming_similarity(self.am, vector)
        return logit

class AssociativeMemoryT():
    """
    Implements an Associative Memory that quantizes MAP to binary using
    Thermometer enconding.
    """
    def __init__(self, intervals: int, bits: int, quantile=0.25):
        super().__init__()
        self.intervals = intervals
        self.bits = bits
        self.quantile = quantile
        self.possible_encodings = bits + 1
        self.vsa = 'BSC'

        if intervals >= self.possible_encodings:
            raise RuntimeError(f'Number of intervals ({intervals}) is to big for the number of bits given ({bits}). Only {self.possible_encodings} are possible.')

        # Create encodings
        #self.enc_table = self.encode_table(self.bits, self.intervals)
        self.enc_table = self.encode_table(self.bits, self.intervals)
        self.poles = None

    def encode_table(self, bits, entries):
        """
        Create the encode table based on a Thermometer encoding.
        """
        all_combinations = bits+1
        t = torchhd.thermometer(all_combinations, bits, vsa='BSC')

        enc_table = t
        # Remove middle entries if the number of required entries is less than
        # the number of thermometer encodings available.
        if entries < all_combinations:
            rows = entries//2 # Number of top and below rows used
            top_rows = t[..., 0:rows, :]
            bot_rows = t[..., -rows:, :]

            # Create the final table without the rows in the middle
            enc_table = torch.vstack([top_rows, bot_rows])
        # enc_table is a BSC vector and it cannot be used together with MAP,
        # otherwise torchhd throws an exception when indexing. Convert the
        # generated themometer table to MAP.
        enc_table = torchhd.ensure_vsa_tensor(enc_table, vsa='MAP', dtype=torch.int32)
        return enc_table

    def _get_poles(self, input: torch.Tensor):
        """
        Return the poles tensor used by this AM. Creates the quantization poles
        values if they do not exist yet based on the given input. Assumes the
        input is normalized.
        """
        if self.poles is None:
            self.poles = _create_poles(input, self.intervals, self.quantile)
        return self.poles

        # Alternative approach generating the quantizaiton poles to each input
        #return self._create_poles(input, self.quantile)

    def transform(self, input):
        t = torchhd.functional.ensure_vsa_tensor(input, vsa='MAP', dtype=torch.float)
        dtype = torch.float

        norm = normalize(t, dtype=dtype)

        poles = self._get_poles(norm)
        inds = _find_poles(norm, poles)

        # Create plain BSC vectors
        res = _map_to_binary_encoding(norm, inds, self.enc_table)
        return res

    def create_am(self, tensors):
        self.am = self.transform(tensors)

    def search(self, query):
        vector = self.transform(query)
        logit = torchhd.hamming_similarity(self.am, vector)
        return logit


class AssociativeMemoryQ():
    """
    Implements an Associative Memory that quantizes the MAP vectors to a linspace.
    """
    def __init__(self, intervals: int):
        super().__init__()
        self.intervals = intervals
        self.vsa = 'MAP'

    def transform(self, input):
        dtype = torch.float
        t = torchhd.functional.ensure_vsa_tensor(input, vsa='MAP', dtype=dtype)

        # Normalize input
        norm = normalize(t, dtype=dtype)

        # Find bit patterns
        # Create 1D tensor with quantization poles
        #poles = torch.linspace(-1, 1, steps=self.intervals)
        poles = torch.linspace(-0.03, 0.03, steps=self.intervals)
        # Make poles a tensor of [Intervals, 1, 1 ..., 1] depending on the
        # shape of the input.
        view = [1] *len(norm.shape)
        poles = poles.view(-1, *view)
        # Get absolute difference to each pole
        diffs = torch.abs(norm - poles)
        _, inds = torch.min(diffs, dim=0)

        # Create plain BSC vectors
        dim = input.shape[-1]
        # Reshape poles to index inds
        poles = poles.view(poles.shape[0])
        res = poles[inds].view((-1, dim))
        return res

    def create_am(self, tensors):
        self.am = self.transform(tensors)

    def search(self, query):
        vector = self.transform(query)
        logit = torchhd.cosine_similarity(self.am, vector)
        return logit

