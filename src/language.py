#!/usr/bin/python3

import argparse
from functools import partial
from itertools import cycle, islice
from pathlib import Path
import torch
from torch import nn
import torchhd
from torchhd.datasets.european_languages import EuropeanLanguages
import torchhd.embeddings as embeddings
from torchhd.types import VSAOptions

import common

MAX_INPUT_SIZE = 128
PADDING_IDX = 0
ASCII_A = ord("a")
ASCII_Z = ord("z")
ASCII_SPACE = ord(" ")
NUM_TOKENS = ASCII_Z - ASCII_A + 3 # "a" through "z" plus space and padding

def char2int(char: str) -> int:
    """Map a character to its integer identifier"""
    ascii_index = ord(char)

    if ascii_index == ASCII_SPACE:
        # Remap the space character to come after "z"
        return ASCII_Z - ASCII_A + 1

    return ascii_index - ASCII_A

def transform(x: str) -> torch.Tensor:
    char_ids = x[:MAX_INPUT_SIZE]

    #Padding is mapped to 0, "a" to 1, "z" to 26, and space to 27
    char_ids = [char2int(char) + 1 for char in char_ids.lower()]

    # Do not append empty HDC vectors to the end of strings that are less than
    # MAX_INPUT_SIZE. The appending does not work for BSC vectors since it
    # biases the bundle operation. Since this code was copied fro torchhd
    # example, the author's intention might be to speed up batch operations,
    # even though I have not noticed any noticeable performance difference.
    #if len(char_ids) < MAX_INPUT_SIZE:
    #    char_ids += [PADDING_IDX] * (MAX_INPUT_SIZE - len(char_ids))

    # Instead, replicate text string until it reaches MAX_INPUT_SIZE.
    if len(char_ids) < MAX_INPUT_SIZE:
        char_ids = list(islice(cycle(char_ids), MAX_INPUT_SIZE))

    return torch.tensor(char_ids, dtype=torch.long)

def load_dataset(dataset_dir, batch_size=1):
    """
    Load EuropeanLanguages dataset
    """
    train_ld, test_ld = common.load_dataset(
            dataset_dir,
            EuropeanLanguages,
            batch_size=batch_size,
            transform=transform)

    return train_ld, test_ld

class Language_HDC(nn.Module):
    """docstring for Encoder"""
    def __init__(
            self,
            dimensions,
            num_classes,
            id_mem_size,
            *,
            vsa: VSAOptions = 'MAP',
            dtype_enc=torch.get_default_dtype(),
            dtype_am=torch.get_default_dtype(),
            **kwargs
        ):
        super(Language_HDC, self).__init__()
        self.dimensions = dimensions
        self.dtype_enc = dtype_enc
        self.dtype_am = dtype_am
        self.vsa: VSAOptions = vsa
        self.id = embeddings.Random(id_mem_size, dimensions, padding_idx=PADDING_IDX, vsa=self.vsa, dtype=self.dtype_enc)

        self.am = common.pick_am_model(vsa, dimensions, num_classes, **kwargs)

    def create_am(self):
        self.am.train_am()

    def encode(self, x: torch.Tensor):
        sample_hv = self.id(x)
        sample_hv = torchhd.ngrams(sample_hv)
        return sample_hv

    def forward(self, x):
        enc = self.encode(x)
        return self.am.search(enc)

def main():
    # Enable parser #
    parser = argparse.ArgumentParser(description='Train the VoiceHD HDC model.')
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

    if args.redirect_stdout:
        log_path = model_path.removesuffix('.pt')+'.log'
        common.redirect_stdout(log_path)

    common.set_random_seed(args.seed)
    device = common.set_device(args.device)

    # Pytorch batch size
    BATCH_SIZE = 1
    train_ld, test_ld = load_dataset(args.dataset_dir, BATCH_SIZE)

    num_classes = len(train_ld.dataset.classes)
    id_mem_size = NUM_TOKENS # One entry in item memory for each encoded token
    constructor = partial(
            Language_HDC,
            args.vector_size,
            num_classes,
            id_mem_size,
            vsa=vsa,
            dtype_enc=dtype_enc,
            dtype_am=dtype_am,
            **am_args,
            )
    model = common.args_pick_model(args, constructor)

    model = common.args_train_hdc(args, model, train_ld)
    accuracy = common.args_test_hdc(args, model, test_ld, num_classes)

    common.save_results(args, model, train_ld, accuracy)

if __name__ == '__main__':
    main()

