#!/usr/bin/python3

# Based on the implementation available in torchhd emg_hand_gestures.py with
# minor modifications. The example implements only the spatiotemporal encoder.

import argparse
from statistics import mean
from pathlib import Path
import torch
from torch import nn
import torch.utils.data as data
import torchhd
from torchhd.datasets import EMGHandGestures
import torchhd.embeddings as embeddings
from torchhd.types import VSAOptions

import common

NUM_LEVELS = 21
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
WINDOW = 256
N_GRAM_SIZE = 4
DOWNSAMPLE = 5
SUBSAMPLES = torch.arange(0, WINDOW, int(WINDOW / DOWNSAMPLE))

def transform(x):
    return x[SUBSAMPLES]

def load_dataset(dataset_dir, subjects=[0], batch_size=1):
    '''
    Load EMGHandGestures dataset.
    '''
    device = common.get_device()
    generator = torch.Generator(device=device)
    ds = EMGHandGestures(
        dataset_dir, download=True, subjects=subjects, transform=transform
    )

    train_size = int(len(ds) * 0.7)
    test_size = len(ds) - train_size
    train_ds, test_ds = data.random_split(
            ds,
            [train_size, test_size],
            generator=generator)

    train_ld = data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=generator)
    test_ld = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_ld, test_ld

class EMG_HDC(nn.Module):
    """docstring for Encoder"""
    def __init__(
            self,
            dimensions,
            timestamps,
            channels,
            levels,
            *,
            vsa: VSAOptions = 'MAP',
            dtype_enc=torch.get_default_dtype(),
            dtype_am=torch.get_default_dtype(),
            **kwargs
        ):
        super(EMG_HDC, self).__init__()
        self.dimensions = dimensions
        self.dtype_enc = dtype_enc
        self.dtype_am = dtype_am
        self.vsa: VSAOptions = vsa
        self.channels = embeddings.Random(channels, dimensions, vsa=self.vsa, dtype=self.dtype_enc)
        # I have no idea why the original example use this timestamps Item
        # Memory since the paper record timestamps by permutating them.
        # Removing it improved accuracy and the results are now closer to
        # Figure 7 presented in the original paper.
        #self.timestamps = embeddings.Random(timestamps, dimensions)
        self.signals = embeddings.Level(levels, dimensions, low=0, high=20, vsa=self.vsa, dtype=self.dtype_enc)

        num_classes = 5
        self.am = common.pick_am_model(vsa, dimensions, num_classes, **kwargs)

    def create_am(self):
        self.am.train_am()

    def encode(self, x):
        # Copied from torchhd's example but removed bind to timestamp
        signal = self.signals(x)
        samples = torchhd.bind(signal, self.channels.weight.unsqueeze(0))
        #samples = torchhd.bind(signal, self.timestamps.weight.unsqueeze(1))

        samples = torchhd.multiset(samples)
        sample_hv = torchhd.ngrams(samples, n=N_GRAM_SIZE)
        return sample_hv

    def forward(self, x):
        enc = self.encode(x)
        return self.am.search(enc)

def experiment(args, am_args, subjects=[0], device="cpu"):
    vsa = args.vsa
    if vsa != "BSC":
        dtype_enc = common.map_dtype(args.dtype_enc)
        dtype_am = common.map_dtype(args.dtype_am)
        model_name = f'sub{subjects[0]}-{vsa.lower()}-enc{args.dtype_enc}-am{args.dtype_am}-d{args.vector_size}.pt'
    else:
        dtype_enc = torch.bool
        dtype_am = torch.bool
        model_name = f'sub{subjects[0]}-{vsa.lower()}-d{args.vector_size}.pt'

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir+'/'+model_name

    if args.redirect_stdout:
        log_path = model_path.removesuffix('.pt')+'.log'
        common.redirect_stdout(log_path)

    train_ld, test_ld = load_dataset(args.dataset_dir, subjects, BATCH_SIZE)

    model = EMG_HDC(
            args.vector_size,
            6,
            4,
            NUM_LEVELS,
            vsa=args.vsa,
            dtype_enc=dtype_enc,
            dtype_am=dtype_am,
            **am_args
            )
    num_classes = len(train_ld.dataset.dataset.classes)
    model = common.train_hdc(
            model,
            train_ld,
            device,
            test_ld=test_ld,
            retrain_rounds=args.retrain_rounds,
            retrain_best=args.retrain_best)
    accuracy = common.test_hdc(model, test_ld, num_classes, device)

    if args.export:
        common.export_model(model, args.export, train_ld)
    if args.save_model:
        common.save_model(model, args.save_model)

    return accuracy

def main():
    # Enable parser #
    parser = argparse.ArgumentParser(description='Train the EMG HDC model.')
    common.add_default_arguments(parser)
    common.add_default_hdc_arguments(parser)
    common.add_am_arguments(parser)

    default_subjects = range(0, 5)
    parser.add_argument('--subject',
            choices=default_subjects,
            default=None,
            type=int,
            help=f'Select the used subject among the available options. If none'
                'give, then execute EMG on all subjects and the final exported '
                'accuracy is the mean of all accuracies obtained.'
            )

    args = parser.parse_args()
    am_args = common.parse_am_group(parser, args)

    common.set_random_seed(args.seed)
    device = common.set_device(args.device)

    subjects = [args.subject] if args.subject is not None else default_subjects
    accuracies = []
    for s in subjects:
        acc = experiment(args, am_args, [s], device=device)
        accuracies.append(acc)

    final_acc = mean(accuracies)
    if len(subjects) > 1:
        print(f'Final mean accuracy {final_acc}%')

    # Print accuracy to file
    if args.accuracy_file:
        common.dump_accuracy(args.accuracy_file, final_acc)

if __name__ == '__main__':
    main()

