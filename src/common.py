import argparse
from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Callable, Optional
from am.am import AMMap, AMSignQuantize, AMThermometerDeviation, PQHDC
import torch
import torchmetrics
from tqdm import tqdm
from patchmodel import transform_am

from am.learning import Centroid
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

    parser.add_argument('--patch-model',
            default=False,
            action='store_true',
            help='Patch a loaded model.')

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
    vsa_type = { 'MAP' }
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
    am_types = ['V', 'TQHD', 'SQ', 'PQHDC']

    group = parser.add_argument_group('Associative Memory', 'Allows fine control of AM parameters.')

    group.add_argument(
            '--am-type',
            choices=am_types,
            default=default_am,
            help='Chooses the AM type used. V is the regular AM, TQHD is the thermometer quantized, SQ is sig quantized, and PQHDC is projection based.'
            )

    default_bits = 8
    group.add_argument(
            '--am-bits',
            default=default_bits,
            type=int,
            help=f'Choose the number of bits expansion used in TQHD. Defaults to {default_bits}.'
            )

    default_intervals = default_bits+1
    group.add_argument(
            '--am-intervals',
            default=default_intervals,
            type=int,
            help=f'Choose the number of intervals used with TQHD AMs. This argument is ignored for other AM types. Defaults to {default_intervals}'
            )

    default_deviation = 1.0
    group.add_argument(
            '--am-tqhd-deviation',
            default=default_deviation,
            type=_float_range(0.0, float('inf')),
            help=f'Choose the multiplier used with the standard deviation when assigning the quantization poles of TQHD. Only positve float values are accepted. Defaults to {default_deviation}.'
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
    learning = Centroid()

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

    learning = Centroid()

    if am_prediction == 'Normal':
        prediction = Normal()
    elif am_prediction == 'Fault':
        prediction = Fault(kwargs['am_fault_rate'])

    if am_type == 'V':
        if vsa == 'MAP':
            am = AMMap(dim, num_classes, learning=learning, **kwargs)
    elif am_type == 'TQHD':
        bits = kwargs['am_bits']
        intervals = kwargs['am_intervals']
        deviation = kwargs['am_tqhd_deviation']
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
        if args.patch_model:
            model = transform_am(args, model)

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
