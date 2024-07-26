#!/usr/bin/python3

import argparse
from functools import partial
from pathlib import Path
import torch
from torch import nn
import torchvision
from torchvision.datasets import MNIST
import torchhd
import torchhd.embeddings as embeddings
from torchhd.types import VSAOptions

import common

def load_dataset(dataset_dir: str, batch_size=1):
    transform = torchvision.transforms.ToTensor()
    train_ld, test_ld = common.load_dataset(dataset_dir, MNIST, batch_size=batch_size, transform=transform)
    return train_ld, test_ld

class Mnist_HDC(nn.Module):
    """docstring for Encoder"""
    def __init__(
            self,
            dimensions,
            image_size,
            levels,
            *,
            vsa: VSAOptions = 'MAP',
            dtype_enc=torch.get_default_dtype(),
            dtype_am=torch.get_default_dtype(),
            **kwargs
        ):
        super(Mnist_HDC, self).__init__()
        self.dimensions = dimensions
        self.dtype_enc = dtype_enc
        self.dtype_am = dtype_am
        self.vsa: VSAOptions = vsa

        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(image_size, dimensions, vsa=self.vsa, dtype=self.dtype_enc)
        self.value = embeddings.Thermometer(levels, dimensions, vsa=self.vsa, dtype=self.dtype_enc)

        num_classes = 10
        self.am = common.pick_am_model(vsa, dimensions, num_classes, **kwargs)

    def create_am(self):
        self.am.train_am()

    def encode(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return sample_hv

    def forward(self, x):
        enc = self.encode(x)
        return self.am.search(enc)

def main():
    # Enable parser #
    parser = argparse.ArgumentParser(description='Train MNIST HDC model.')
    common.add_default_arguments(parser)
    common.add_default_hdc_arguments(parser)
    common.add_am_arguments(parser)

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

    if vsa == 'FHRR':
        dtype_enc = torch.complex64
        dtype_am = torch.complex64

    if args.redirect_stdout:
        log_path = model_path.removesuffix('.pt')+'.log'
        common.redirect_stdout(log_path)

    common.set_random_seed(args.seed)
    device = common.set_device(args.device)
    # Pytorch batch size
    BATCH_SIZE = 1
    # HDC Model variables
    LEVELS = 10

    train_ld, test_ld = load_dataset(args.dataset_dir, batch_size=BATCH_SIZE)

    num_classes = len(train_ld.dataset.classes)
    IMG_SIZE = 28
    constructor = partial(
            Mnist_HDC,
            args.vector_size,
            IMG_SIZE*IMG_SIZE,
            LEVELS,
            vsa=vsa,
            dtype_enc=dtype_enc,
            dtype_am=dtype_am,
            **am_args
            )
    model = common.args_pick_model(args, constructor)

    model = common.args_train_hdc(args, model, train_ld, test_ld=test_ld)
    accuracy = common.args_test_hdc(args, model, test_ld, num_classes)

    common.save_results(args, model, train_ld, accuracy)

if __name__ == '__main__':
    main()

