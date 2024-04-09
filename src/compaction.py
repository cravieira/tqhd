#!/usr/bin/python3

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

    return [*stats_0, *stats_1], [*headers_0, *headers_1]

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
    '''
    Count the number of contiguous elements in the given tensor.
    '''
    count_1D = partial(torch.unique_consecutive, return_counts=True)
    ret1, ret2 = tee(map(count_1D, t))
    outputs, counts = [e[0] for e in ret1], [e[1] for e in ret2]

    #outputs, counts = torch.unique_consecutive(t, dim=1, return_counts=True)
    return outputs, counts

def compact_vector(vector, bits, compaction_bits, min_val=0):
    compacted = []
    #min_val = min_val
    max_val = (compaction_bits**2)-1+min_val
    for elem in vector:
        if elem <= max_val:
            compacted.append(elem.item())
        else:
            left = elem.item()
            while left > 0:
                bits_compacted = min(left,max_val)
                compacted.append(bits_compacted)
                compacted.append(min_val)
                left -= bits_compacted

    return compacted

def compact_am(am, bits, compaction_bits, min_val=0):
    '''
    Returns the number of elements in the compacted AM.
    '''
    _, counts = count_contiguous(am)
    f = partial(compact_vector, bits=bits, compaction_bits=compaction_bits, min_val=min_val, )
    # Compacted vectors has shape [<am classes>, <variable>]. Each 1D array in
    # compacted_vectors is an array of numbers, where each number is the length
    # of a symbol.
    compacted_vectors = list(map(f, counts))

    original_size = am.numel()

    # Get the number of elements in each compacted vector
    compacted_sizes = list(map(len, compacted_vectors))
    # Get the number of elements in the compacted list
    # Accumulate the sizes
    acc = lambda a, b: a+b
    compacted_elements = reduce(acc, compacted_sizes)
    # The size in bits of a compacted AM is the number of elements in the
    # compacted AM * the size of each element in bits.
    compacted_size = compacted_elements * compaction_bits

    #print(f'Original size: {original_size}')
    ##print(f'Compacted size {compaction_bits}*{len(compacted)}:', compacted_size)
    #print(f'Compacted size:', compacted_size)
    #improvement = ((original_size/compacted_size)-1)*100
    #print(f'Improvement: {improvement:.2f}%')

    return compacted_elements

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
            '-c',
            '--compaction-bits',
            required=True,
            type=int,
            help='Number of bits used in compaction.',
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

    c_bits = args.compaction_bits

    model = common.load_model(args.model)
    if args.patch_model:
        model = transform_am(args, model)

    am = get_am_tensor(model)
    bits = get_am_bits(model)
    am_classes = get_am_classes(model)
    subject_am = am
    if not args.no_interleave:
        subject_am = flip_blocks(am, bits)
    c_elem = compact_am(subject_am, bits, c_bits, min_val=0)

    stats, stats_headers = get_statiscs(subject_am)

    # Dimensions in the unquantized MAP model
    map_dim = am.shape[-1]/bits
    tqhd_am_size = am_classes*bits*map_dim
    comp_am_size = c_elem * c_bits
    header = [
            'am_classes', # Number of classes for this application
            'map_dim', # Number of dimensions in the unquantized AM
            'tqhd_b', # Number of bits used TQHD quantization
            'tqhd_am_size', # Size of uncompressed TQHD AM in bits
            'comp_c', # Number of bits used in compacted words
            'comp_elems', # Number of elements in compacted AM
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
            am_classes,
            map_dim,
            bits,
            tqhd_am_size,
            c_bits,
            c_elem,
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
