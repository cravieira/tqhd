#!/usr/bin/python3

'''
This script serializes the datasets used by the models in ./src to plain torch
tensors so that they can be deployed to the C++ environment. This script relies
that each model implemented knows how to load its own dataset. This way, it is
possible to make the serialization loosely coupled to each dataset.
'''

import argparse
import torch
from pathlib import Path

def _gettensors(data_loader):
    '''
    Returns the data and target tensors of a given DataLoader. This function
    applies any operation made by the DataLoader, such as transform or shuffle.
    '''
    # Get all entries from the DataLoader so it can apply any transformation it
    # may have.
    data_targets = [i for i in data_loader]
    # data_targets is a list of lists of tensors in the form [[data, target]].
    # Split the data from the target. Each new variable, data and targets, are
    # a list of tensors.
    data = [data for data, _ in data_targets]
    targets = [target for _, target in data_targets]
    # Transform from list to torch.tensor
    data = torch.stack(data)
    targets = torch.stack(targets)

    return data, targets

def save_tensor(tensor, path):
    """Save tensor to path. Create parent directories if necessary."""
    p = Path(path)
    Path.mkdir(p.parent, parents=True, exist_ok=True)
    torch.save(tensor, path)

def voicehd(dataset_dir, serial_dir):
    out_dir = serial_dir+"/voicehd"

    from voicehd_hdc import load_dataset
    _, test_ld = load_dataset(dataset_dir)
    data, targets = _gettensors(test_ld)
    save_tensor(data, out_dir+'/test_data.pt')
    save_tensor(targets, out_dir+'/test_label.pt')

def mnist(dataset_dir, serial_dir):
    out_dir = serial_dir+"/mnist"

    from mnist_hdc import load_dataset
    _, test_ld = load_dataset(dataset_dir)
    data, targets = _gettensors(test_ld)
    save_tensor(data, out_dir+'/hdc/test_data.pt')
    save_tensor(targets, out_dir+'/hdc/test_label.pt')

    from mnist_lenet import load_dataset
    _, test_ld = load_dataset(dataset_dir)
    data, targets = _gettensors(test_ld)
    save_tensor(data, out_dir+'/lenet/test_data.pt')
    save_tensor(targets, out_dir+'/lenet/test_label.pt')

def language(dataset_dir, serial_dir):
    out_dir = serial_dir+"/language"

    from language import load_dataset
    _, test_ld = load_dataset(dataset_dir)
    data, targets = _gettensors(test_ld)
    save_tensor(data, out_dir+'/test_data.pt')
    save_tensor(targets, out_dir+'/test_label.pt')

def emg(dataset_dir, serial_dir):
    out_dir = serial_dir+"/emg"

    from emg import load_dataset
    for i in range(0, 4):
        _, test_ld = load_dataset(dataset_dir, subjects=[i])
        data, targets = _gettensors(test_ld)
        save_tensor(data, out_dir+f'/test_data{i}.pt')
        save_tensor(targets, out_dir+f'/test_label{i}.pt')

def main():
    default_dataset_dir = './_data'
    default_serial_dir = './_serial'

    parser = argparse.ArgumentParser(description='Serialize Pytorch datasets to files.')
    parser.add_argument(
            '-d', '--dataset-dir',
            default=default_dataset_dir,
            help=f'Dataset prefix path. This path contains the downloaded\
            datasets. Notice that a new folder will be created inside the given\
            path to store the dataset. The prefix path is created if necessary.\
            Defaults to \"{default_dataset_dir}\".')
    parser.add_argument(
            '-o', '--output-dir',
            default=default_serial_dir,
            help=f'Output directory with serialized datasets. Parent folders\
            are created if necessary. Defaults to \"{default_serial_dir}\".')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)
    dataset_dir = args.dataset_dir
    serial_dir = args.output_dir

    Path.mkdir(Path(serial_dir), parents=True, exist_ok=True)
    voicehd(dataset_dir, serial_dir)
    mnist(dataset_dir, serial_dir)
    language(dataset_dir, serial_dir)
    emg(dataset_dir, serial_dir)

if __name__ == '__main__':
    main()
