#!/usr/bin/python3

'''
During research, we considered using a simple Run-length encoding (RLE) method
for TQHD's hypervector compression. However, the results perfomed slightly
worse than the final compression approach shown in the paper. Therefore, the
source code presented here is not used in the final work. This file is kept in
the repository for legacy purposes.
'''

import torch
import common
from hdc_models import *
from functools import partial, reduce
from itertools import tee
import argparse
import numpy as np
import csv
from pathlib import Path

from patchmodel import transform_am

def get_am_tensor(model):
    return model.am.am

def get_am_bits(model):
    return model.am.bits

def get_am_classes(model):
    return model.am.num_classes

def get_mean(t: torch.Tensor):
    return torch.mean(t, dtype=torch.float)

def std(t: torch.Tensor):
    return torch.std(t)

def get_statiscs(am: torch.Tensor):
    # Count the number of contiguous elements
    _, counts = count_contiguous(am)
    # Transform Pytorch tensors to numpy arrays
    counts = [c.numpy(force=True) for c in counts ]

    # Compute the mean number of groups in an AM. This is usefuld to evaluate
    # the fragmentation of a encoding/interleaving combination. Consider the
    # following binary vector:
    # 0001001110110
    # Its "counts" of contiguous groups is [3, 1, 2, 3, 1, 2, 1], and its
    # number of groups is 7.
    number_of_groups = np.array(list(map(len, counts)))
    mean_groups = np.mean(number_of_groups)

    # Split the group lengths between 0s and 1s. This is useful to evaluate
    # better compression on encoding strategies that have a different number of
    # 0s and 1s such as BandMatrix.
    def _get_count_of_element(elem, counts, am):
        elem_counts = []
        for c, class_hv in zip(counts, am):
            first_element = 0
            if class_hv[0] != elem:
                first_element = 1

            desired_count = c[first_element::2]
            elem_counts.append(desired_count)

        return elem_counts
    counts_0 = _get_count_of_element(0, counts, am)
    counts_1 = _get_count_of_element(1, counts, am)

    headers = [
        'mean',
        'std',
        'min',
        'q1',
        'median',
        'q3',
        'max',
        ]
    def _get_statistics(counts):
        np_q1 = partial(np.quantile, q=0.25)
        np_q3 = partial(np.quantile, q=0.75)

        means = [*map(np.mean, counts)]
        stds = [*map(np.std, counts)]
        mins = [*map(np.min, counts)]
        q1s = [*map(np_q1, counts)]
        medians = [*map(np.median, counts)]
        q3s = [*map(np_q3, counts)]
        maxs = [*map(np.max, counts)]

        mean = np.mean(means)
        std = np.mean(stds)
        min = np.mean(mins)
        q1 = np.mean(q1s)
        median = np.mean(medians)
        q3 = np.mean(q3s)
        max = np.mean(maxs)

        return mean, std, min, q1, median, q3, max
    stats_0 = _get_statistics(counts_0)
    stats_1 = _get_statistics(counts_1)
    headers_0 = [h+'_0' for h in headers]
    headers_1 = [h+'_1' for h in headers]

    return [mean_groups, *stats_0, *stats_1], ['mean_groups', *headers_0, *headers_1]

def print_statistics(t: torch.Tensor):
    print('mean:', torch.mean(t, dtype=torch.float))
    print('mode:', torch.mode(t)[0].item())
    print('std:', torch.std(t.to(torch.float)))

def flip_blocks(t: torch.Tensor, block_size: int):
    '''
    Split the data in the last axis into blocks and flip the second block in a
    pair. Example for block = 4:
        [1,1,0,0,1,1,0,0]
    Result:
        [1,1,0,0,0,0,1,1]
    '''
    # Check if block_size is multiple of tensor size
    if (t.shape[-1] % block_size) != 0:
        raise RuntimeError('Last size in shape is not multiple of block size.')

    split_blocks = t.view(t.shape[0], -1, 2, block_size)
    even_blocks = split_blocks[:,:,1]
    # Flip each block, i.e., flip each element in the last axis.
    flipped_blocks = torch.flip(even_blocks, [2])
    ret = torch.clone(split_blocks)
    ret[:,:,1] = flipped_blocks
    # Return to original shape
    ret = ret.view(t.shape)
    return ret

def count_contiguous(t: torch.Tensor):
    """
    Count the number of contiguous elements in the given 2D tensor along dim=1.

    :param t: A 2D tensor.
    """
    count_1D = partial(torch.unique_consecutive, return_counts=True)
    ret1, ret2 = tee(map(count_1D, t))
    outputs, counts = [e[0] for e in ret1], [e[1] for e in ret2]

    #outputs, counts = torch.unique_consecutive(t, dim=1, return_counts=True)
    return outputs, counts

# Return the size of a RLE compression of vectors containing only two symbols.
# It is possible to specify a different number of bits for each symbol.
def binary_rle(
        vector,
        symbol_0_size: int,
        symbol_1_size: int,
        first_symbol: int,
        min_val: int=0
    ):
    """
    Given a vector of lengths of only two symbols, i.e., "0" and "1", compute
    the run length encoding considering each symbol size.

    :param vector [TODO:type]: [TODO:description]
    :param symbol_0_size: Number of bits for each symbol "0". Must be > 0.
    :param symbol_1_size: Number of bits for each symbol "1". Must be > 0.
    :param first_symbol: The first symbol in the vector of lengths, either "0" or "1".
    :param min_val: Placeholder variable for min lengths. This argument is currently ignored.
    :raises RuntimeError: Raises an exception if "symbol_0_size" or "symbol_1_size" are <= 0.
    """
    if symbol_0_size <= 0 or symbol_1_size <=0:
        raise RuntimeError(f'Symbol sizes must be at least 1. Given: {symbol_0_size} and {symbol_1_size}')

    compacted = []
    max_0_val = (2**symbol_0_size)-1+min_val
    max_1_val = (2**symbol_1_size)-1+min_val

    # Set variables according to the first element
    current_symbol = 0
    current_size = symbol_0_size
    max_val = max_0_val
    if first_symbol:
        current_symbol = 1
        current_size = symbol_1_size
        max_val = max_1_val

    def _swap_symbol():
        """
        Swap the current symbol being analyzed by swapping the outer function's
        variables.
        """
        nonlocal current_symbol, current_size, symbol_0_size, symbol_1_size
        if current_symbol:
            current_symbol = 0
            current_size = symbol_0_size
            max_val = max_0_val
        else:
            current_symbol = 1
            current_size = symbol_1_size
            max_val = max_1_val

        return current_symbol, current_size, max_val

    for elem in vector:
        length = elem.item()
        if elem <= max_val:
            compacted.append(length)

            current_symbol, current_size, max_val = _swap_symbol()
        else:
            left = length
            while left > 0:
                bits_compacted = min(left, max_val)
                compacted.append(bits_compacted)
                current_symbol, current_size, max_val = _swap_symbol()

                left -= bits_compacted
                # Check if the last part of the length was compressed
                if left <= 0:
                    break
                compacted.append(min_val)
                current_symbol, current_size, max_val = _swap_symbol()

    return compacted

def compress(hyper_vector, symbol_0_size, symbol_1_size, min_val=0):
    """docstring for compress"""
    hv = hyper_vector.reshape(1, -1)
    _, contiguous = count_contiguous(hv)
    contiguous = contiguous[0]
    first_element = hyper_vector[0]
    rle_lengths = binary_rle(contiguous, symbol_0_size, symbol_1_size, first_symbol=first_element, min_val=min_val)

    # Compute how many bits are necessary for each symbol
    rle_lengths = np.array(rle_lengths)
    if first_element == 0:
        lengths_0 = rle_lengths[0::2]
        lengths_1 = rle_lengths[1::2]
    else:
        lengths_0 = rle_lengths[1::2]
        lengths_1 = rle_lengths[0::2]

    sizes_0 = len(lengths_0) * symbol_0_size
    sizes_1 = len(lengths_1) * symbol_1_size
    total_size = sizes_0 + sizes_1
    return total_size

def compact_am(am, symbol_0_size, symbol_1_size, min_val=0):
    '''
    Returns the number of elements in the compacted AM.
    '''
    f = partial(compress, symbol_0_size=symbol_0_size, symbol_1_size=symbol_1_size, min_val=min_val, )
    # Compacted vectors has shape [<am classes>, total size]. Each 1D array in
    # compacted_vectors is the size of the compressed vector
    compacted_sizes = list(map(f, am))

    # Accumulate the sizes
    acc = lambda a, b: a+b
    compacted_size = reduce(acc, compacted_sizes)
    return compacted_size

def main():
    parser = argparse.ArgumentParser(
        description='Analyze and compact a serialized TQHD model. This script '
        'also accepts MAP models and can patch it to TQHD before running using '
        '"--patch-model".')
    parser.add_argument(
            'model',
            help='Path to model.',
        )

    parser.add_argument('--patch-model',
            default=False,
            action='store_true',
            help='Patch the loaded model before compaction.')

    parser.add_argument(
            '-c0',
            '--compaction-bits0',
            required=True,
            type=int,
            help='Number of bits used in compaction of symbol 0.',
            )
    parser.add_argument(
            '-c1',
            '--compaction-bits1',
            required=True,
            type=int,
            help='Number of bits used in compaction of symbol 1.',
            )
    parser.add_argument(
            '--csv',
            type=str,
            help='Print statistics in a comma separated values file in the'
            'given path. Parent directories are created if necessary.'
            )
    parser.add_argument(
            '--no-interleave',
            action='store_true',
            help='Disable vector interleaving when analyzing the AM.'
            )

    common.add_am_arguments(parser)

    args = parser.parse_args()

    interleaving = not args.no_interleave
    encoding_name = args.am_tqhd_encode_table.name

    c0_bits = args.compaction_bits0
    c1_bits = args.compaction_bits1

    model = common.load_model(args.model)
    if args.patch_model:
        model = transform_am(args, model)

    am = get_am_tensor(model)
    bits = get_am_bits(model)
    am_classes = get_am_classes(model)
    subject_am = am
    if interleaving:
        subject_am = flip_blocks(am, bits)
    comp_am_size = compact_am(subject_am, args.compaction_bits0, args.compaction_bits1, min_val=0)

    stats, stats_headers = get_statiscs(subject_am)

    # Dimensions in the unquantized MAP model
    map_dim = am.shape[-1]/bits
    tqhd_am_size = am_classes*bits*map_dim
    #comp_am_size = c_elem * c_bits
    header = [
            'encoding',
            'interleaving',
            'am_classes', # Number of classes for this application
            'map_dim', # Number of dimensions in the unquantized AM
            'tqhd_b', # Number of bits used TQHD quantization
            'tqhd_am_size', # Size of uncompressed TQHD AM in bits
            'compaction_bits_0', # Number of bits used in compacted words for symbol 0
            'compaction_bits_1', # Number of bits used in compacted words for symbol 1
            #'comp_elems', # Number of elements in compacted AM
            'comp_am_size', # Size of compressed AM in bits
            # Bit groups statistics. Each vector in a PQHD AM is formed by
            # thermometer encoded words which have contiguous groups of 0s and
            # 1s. The exported CSV contains statistics considering the length
            # of these groups. The length stats are dumped for each symbol, 0
            # and 1.
            # Bit group length stats:
            #'mean', # Mean length of bit groups
            #'std', # Standard deviation
            #'min', # Min length of a bit group
            #'q1', # First quantile
            #'median', # Median of the length
            #'q3', # Third quantile
            #'max' # Max length of a bit group
            *stats_headers
            ]
    data = [
            encoding_name,
            interleaving,
            am_classes,
            map_dim,
            bits,
            tqhd_am_size,
            c0_bits,
            c1_bits,
            #c_elem,
            comp_am_size,
            *stats]
    if len(header) != len(data):
        raise RuntimeError(f'Missmatch of sizes between Data ({len(data)}) and Header ({len(header)}). Check this script code.')
    if args.csv:
        path = Path(args.csv)
        common.create_path(path.parent)
        with open(args.csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)
    else:
        print(*header)
        print(*data)

if __name__ == '__main__':
    main()

