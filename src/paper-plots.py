#!/usr/bin/python3

from cycler import cycler
from functools import partial
import matplotlib.pyplot as plt
from numpy.typing import NDArray
plt.style.use('ggplot')
#plt.style.use('seaborn-v0_8-darkgrid')
import matplotlib.lines as mlines
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib import ticker
import torch
import math
import numpy as np
import natsort
from pathlib import Path
import re
import pandas as pd
from typing import List

# Plots imports
from plot import plot

# Models imports
# Hack to solver imports
import sys
sys.path.append("train/src")
import common
import voicehd_hdc as voicehd

# Applications #
GRAPHHD_DATASETS = [
    'graphhd-dd',
    'graphhd-enzymes',
    'graphhd-mutag',
    'graphhd-nci1',
    'graphhd-proteins',
    'graphhd-ptc_fm',
]

# App names as they appear in plots
APP_PLOT_NAMES = [
    'voicehd',
    'mnist',
    'language',
    'hdchog',
    'emg',
    'graphhd'
]

# App names as they appear in the directory tree
APP_DIR_NAMES = [
    'voicehd',
    'mnist',
    'language',
    'hdchog-fashionmnist',
    'emg-all',
]

APP_NAME_STYLE = dict(
            fontstyle='oblique',
        )

def make_latex_equation(t: str) -> str:
    return '$'+t+'$'

def str_replace(t: str, pattern: str, to: str) -> str:
    return t.replace(pattern, to)

def print_labeled(array, labels: List[str]):
    if len(array) != len(labels):
        raise RuntimeError('Attempt to print labeled data with diferent'
            ' unequal number of labels')
    for val, label in zip(array, labels):
        print(f'{label}: {val}')


def select_bits(data, labels):
    '''
    The functions in this script parse all experiments executed for each
    application. However, it can be nice for the plots to select only a
    fraction of the experiments executed. For example, if TQHD was experimented
    with bits from 2 to 8, then this function can restrict the results to 2 to
    4 so that the plots do not become polluted.
    '''
    new_data = []
    bit_init = 0
    bit_end = None
    if bit_end is None:
        return data, labels
    for app in data:
        new_data.append(app[bit_init:bit_end])

    new_labels = []
    for i in range(bit_init, bit_end):
        new_labels.append(labels[i])

    return new_data, new_labels

def _parse_dim_dir(path: str):
    accs = parse_accuracy_directory(path)
    ## TODO: Quick fix to set all apps to he same number of seeds
    #accs = accs[0:5]
    return accs

def _parse_bit_dir(path: str):
    dims = map_sorted_subfolders(path, _parse_dim_dir)
    dims = list(dims)
    return dims

def _parse_app_dir(path: str):
    bits = list(map_sorted_subfolders(path, _parse_bit_dir))
    bits = np.array(bits)
    return bits

def _parse_app(acc_path: str):
    quantized_accs = _parse_app_dir(acc_path)

    return quantized_accs

def parse_transformation_apps(apps: List[str], transformation_name: str) -> NDArray:
    acc_apps = []
    for app in apps:
        acc_path = f'_transformation/{app}/hdc/{transformation_name}'
        app_acc = _parse_app(acc_path)
        acc_apps.append(app_acc)

    # TODO: remove this later
    # Make sure all apps for the same transformation have the same number of
    # bits/retraining iterations
    lenghts = map(len, acc_apps)
    min_len = min(lenghts)
    acc_apps = [acc[:min_len] for acc in acc_apps]

    acc_apps = np.array(acc_apps)

    return acc_apps

def parse_app_mean(paths: List[str]) -> NDArray:
    """
    Parse a list of application directory paths and return a NDArray with the
    mean accuracies.

    :param paths: A list of application paths.
    :return: A NDArray with the mean of all parsed accuracies.
    """
    accs = list(map(_parse_app, paths))
    accs = np.array(accs)
    mean_acc = np.mean(accs, axis=0)
    return mean_acc

def parse_transformation_emg(transformation_name: str, all=True) -> NDArray:
    app = 'emg'
    subjects = range(5)
    app_acc = None

    # Parse the result of all subjects produced by emg. Each value is
    # already the mean of all subjects and is provided by the EMG script.
    if all:
        app_acc = _parse_app(
                f'_transformation/{app}/hdc/all/{transformation_name}'
                )
    # Parse each subject individually
    else:
        paths = [f'_transformation/{app}-s{i}/hdc/{transformation_name}' for i in subjects]
        app_acc = parse_app_mean(paths)
    return app_acc

def parse_graphhd():
#TODO: Parse graphhd
    pass

def parse_app_reference_dir(path: str) -> NDArray:
    '''
    Parse all map-encf32-amf32 of an application.
    '''
    reference_model = 'map-encf32-amf32'
    reference_path = Path(path)
    reference_models = reference_path.glob(f'./{reference_model}-d*')
    reference_models = natsort.humansorted(reference_models)
    reference_accs = list(map(parse_accuracy_directory, reference_models))
    reference_accs = [acc[0:20] for acc in reference_accs]
    ## TODO: Quick fix to set all apps to he same number of seeds. REMOVE THIS LATER
    #reference_accs = [acc[0:5] for acc in reference_accs]
    reference_accs = np.array(reference_accs)

    return reference_accs

def parse_reference_apps(apps: List[str]) -> NDArray:
    accs = []
    for app in apps:
        reference_acc = parse_app_reference_dir(f'_accuracy/{app}/hdc')
        accs.append(reference_acc)

    accs = np.array(accs)
    return accs

def get_reference_accs():
    """docstring for parse_reference_apps

    Get reference accuracy of all apps used in all plots.
    """
    global APP_DIR_NAMES
    global GRAPHHD_DATASETS

    apps = APP_DIR_NAMES
    acc_ref = parse_reference_apps(apps)
    # Consider the mean of all GraphHD datasets
    graphhd_acc_all = parse_reference_apps(GRAPHHD_DATASETS)
    graphhd_acc = np.mean(graphhd_acc_all, axis=0)
    # acc_ref.shape = [apps, seed]
    acc_ref = np.vstack((acc_ref, [graphhd_acc]))

    return acc_ref

def figure_histogram(data, sty, legends=None, **kwargs):
    # This function is based on the example available on:
    # https://matplotlib.org/stable/gallery/statistics/histogram_features.html

    # Make a grid of 4 axis
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False)

    num_bins = 42

    def _pdf(data, x):
        # Add a 'best fit' line
        mu = np.mean(data)
        sigma = np.std(data)
        temp = math.sqrt(2 * np.pi)*sigma

        y = ((1 / (temp)) *
             np.exp(-0.5 * (1 / sigma * (x - mu))**2))

        return y

    if legends:
        size_leg = len(legends)
        size_data = len(data)
        if size_leg != size_data:
            raise RuntimeError(f'Number of legends ({size_leg}) is different from the number of data list given (size_data).')
    else:
        legends = [None] * len(data)

    loop_iter = zip(axs.reshape(-1), data, sty)

    for ax, x, sty in loop_iter:
        # Add the standard deviation text to each plot
        x_norm = x / np.linalg.norm(x)
        sigma = np.std(x_norm)

        sigma = np.std(x_norm) # Standard deviation of distribution

        # Create x_ticks related to the standard deviation
        ticks = 3 # Number of std deviation ticks to the left and right of 0
        # Define the plot range regarding the x axis
        x_range = (-ticks, ticks)

        # Refactor xticks position
        x_ticks = np.linspace(*x_range, num=ticks*2+1)

        # Create the strings used under each tick on the x axis
        j = -ticks
        x_ticks_labels = []
        while j <= ticks:
            if j == 0:
                label = '0'
            else:
                label = f'{j}'r'$\sigma$'
                #label = f'{j}'
            x_ticks_labels.append(label) # Create latex sigma symbols
            j += 1

        # Scale X data to its std deviation
        x_scaled = x_norm / sigma

        n, bins, patches = ax.hist(x_scaled, num_bins, density=True, range=x_range, edgecolor='w', **sty)

        #plt.xticks(x_ticks, labels=x_ticks_labels)
        y = _pdf(x_scaled, bins)
        ax.plot(
                bins,
                y,
                '--',
                # color 3
                color=plot.lighten_color(sty['color'], 1.1),
                linewidth=3.0,
                path_effects=[
                    path_effects.SimpleLineShadow(
                        offset=(0.2, 0.2),
                        ),
                    path_effects.Normal()
                    ]

                )
        ax.set_xticks(range(x_range[0], x_range[1]+1))
        ax.set_xticklabels(x_ticks_labels)

        ax.text(0.95, 0.9,
                f"$\sigma={sigma:.4f}$",
                ha='right',
                va='top',
                transform=ax.transAxes,
                )

        # Fix y limits of all axis to make them look nicer in the final Figure
        ax.set_ylim(ymax=0.6)

    ax.plot([], [], '--', color='black', label='PDF')
    # Hack to make the legend box not appear in front of the toppest plots. I
    # think matplotlib tight_layout is not working propperly.
    axs.flatten()[0].set_title(' ')
    fig.legend(loc='outside upper center', ncols=len(data)+1, fontsize='x-small')

    fig.supxlabel(r'Value ($\sigma$)')
    fig.supylabel('Probability Density')
    ax.set_axisbelow(True)
    ax.grid(visible=True)
    plt.tight_layout()

    #plt.show()
    plot._plot(**kwargs)

def figure_normal_distribution():
    device = 'cpu'
    common.set_random_seed(0)

    train_ld, _ = voicehd.load_dataset('_data')
    num_classes = len(train_ld.dataset.classes)
    entry_size = len(train_ld.dataset[0][0])
    LEVELS = 10
    vsa = 'MAP'
    dtype_enc=torch.float
    dtype_am=torch.float
    am_type = 'V'
    models = []
    dims = [1000, 10000]
    for dim in dims:
        model = voicehd.VoiceHD_HDC(
                dim,
                LEVELS,
                num_classes,
                entry_size,
                vsa=vsa,
                dtype_enc=dtype_enc,
                dtype_am=dtype_am,
                am_type=am_type,
                )
        model = model.to(device)
        models.append(model)

    # Get a sample query vector encoded by each model
    sample = next(iter(train_ld))[0]

    # Get query vector
    queries = [model.encode(sample) for model in models]
    # Transform from torch [1, DIM] Tensor to plain 1D [DIM]
    queries = [val.view((-1)) for val in queries]

    # Create legends referring to query vectors
    query_legends = [f'Query D{d}' for d in dims]

    # Train models
    for model in models:
        common.train_hdc(model, train_ld, device=device)

    # Get a trained AM vector from each model
    classes = [model.am.am[0] for model in models]
    class_legends = [f'Class D{d}' for d in dims]
    # Order the data to be plotted in a interleaved pattern between queries and class vectors
    data = [val for pair in zip(queries, classes) for val in pair]
    # Transform torch Tensors to plain numpy arrays
    data = [val.numpy(force=True) for val in data]
    legends = [val for pair in zip(query_legends, class_legends) for val in pair]

    #color_cycle = cycler(facecolor=plt.rcParams['axes.prop_cycle'][:4])
    # Custom colors
    #color_cycle = cycler(color=plt.rcParams['axes.prop_cycle'][:4])
    colors = ['#629fca', '#ffa556', '#6bbc6b', '#e26768',]
    color_cycle = cycler(color=colors)
    label_cycle = cycler(label=legends)

    name='probability-density'
    figure_histogram(
            data,
            sty=color_cycle+label_cycle,
            legends=legends,
            path=f'_plots/{name}.pdf'
            )
    figure_histogram(
            data,
            sty=color_cycle+label_cycle,
            legends=legends,
            path=f'_plots/{name}.png'
            )

def parse_accuracy_file(p: Path):
    with open(p) as f:
        text = f.readline()
        return float(text)

def parse_accuracy_directory(p: Path):
    accs_files = p.glob('./*.acc')
    accs_files = natsort.humansorted(list(accs_files))
    accs = map(parse_accuracy_file, accs_files)
    accs = np.array(list(accs))

    return accs

def get_sorted_subfolders(p: Path):
    p = Path(p)
    objs = p.glob('./*')
    dirs = filter(Path.is_dir, objs)
    sorted_dirs = natsort.humansorted(dirs)
    return sorted_dirs

def map_sorted_subfolders(p: Path, f):
    '''
    Apply the function in f to all subdiretories of p.
    '''
    sorted_dirs = get_sorted_subfolders(p)
    return map(f, sorted_dirs)

def map_sorted_subfiles(p: Path, f):
    '''
    Apply the function in f to all subdiretories of p.
    '''
    p = Path(p)
    objs = p.glob('./*')
    files = [obj for obj in objs if obj.is_file]
    sorted_dirs = natsort.humansorted(files)
    return map(f, sorted_dirs)

def extract_substring(regular_exp, string: str) -> str:
    match = regular_exp.search(string)
    if match is None:
        raise RuntimeError(f'Unable to find regular expression pattern'
                f'"{regular_exp.pattern}" in string "{string}".')
    span = match.span()
    begin = span[0]
    end = span[1]
    return string[begin:end]

def plot_deviation(data, dev_range, apps, labels, hline=None, **kwargs):
    """docstring for plot_deviation"""
    ax_size = len(data)
    #if ax_size != 4:
    #    raise RuntimeError('Argument "data" must have 4 entries, one for each application')
    cm = 1/2.54  # centimeters in inches
    IEEE_column_width = 8.89*cm # Column width in IEEE paper in cms
    #fig, axs = plt.subplots(
    #        nrows=ax_size,
    #        ncols=2,
    #        #sharex=True,
    #        figsize=(IEEE_column_width*2, IEEE_column_width*3)
    #        )
    fig, axs = plt.subplots(
            nrows=math.ceil(ax_size/2),
            ncols=2,
            #sharex=True,
            figsize=(IEEE_column_width*2, IEEE_column_width*1.5)
            )
    axs = axs.flatten()

    # If hline given, draw a horizontal line at the y value given by h_line.
    if hline is None:
        hline = [None] * len(data)

    # Adjust xticks to be in multiples of 0.2, and to end exactly at the end of
    # the dev_range parameter. The variables defined below are used in all axis.
    step = 0.2
    xticks_pos = np.arange(step, max(dev_range)+step, step)
    xticks_str = [f'{val:.1f}' for val in xticks_pos]

    for app_data, ax, app_name, hline_data in zip(data, axs, apps, hline):
        min_y = float('inf')
        for curve, label in zip(app_data, labels):
            ax.plot(
                dev_range,
                curve,
                label=label
            )

            # This step is used to align the y axis to 0. This prevents
            # matplotlib to create y axis with different starting points.
            min = np.min(curve)
            min_y = min if min < min_y else min

        # Place application name at the right of each plot
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(
            app_name,
            rotation=270,
            rotation_mode='anchor',
            verticalalignment="bottom",
            **APP_NAME_STYLE,
        )

        # Draw a horizontal line if given
        if hline_data is not None:
            ax.hlines(
                hline_data,
                xmin=np.min(dev_range),
                xmax=np.max(dev_range),
                label='$SQ$',
                color='black',
                linestyle='--',
            )

        ax.autoscale(enable=True, axis='x', tight=True)
        # Check if any plotted data is below 0
        if min_y > 0:
            ax.set_ylim(ymin=np.floor(min_y))

        # Adjust xticks to be in multiples of 0.2 and to end exactly at the end of
        # dev_range parameter.
        ax.set_xticks(xticks_pos)
        ax.set_xticklabels(xticks_str)

    handles, labels = ax.get_legend_handles_labels()
    # Hack to make the legend box not appear in front of the toppest plots. I
    # think matplotlib tight_layout is not working propperly.
    fig.legend(handles, labels, loc='outside upper center', ncols=len(handles)+1, fontsize='small')
    axs[0].set_title(' ')

    fig.supylabel('Accuracy Loss in pp')
    fig.supxlabel(r'$P$ limits scaled to $\sigma$')

    plt.tight_layout()
    plot._plot(**kwargs)

def figure_error_deviation(dim=1000, suplementary=False):
    """
    Plot the error to the baseline when varying the standard deviation used.
    """

    def _parse_app_results(experiment_path, reference_path, sq_path, dim):
        """docstring for _parse_app_results"""
        # Get all result directories that use a given dimension
        all_dirs = list(experiment_path.glob(f'./*-d{dim}'))
        # Find the number of bits used in the experiment. The directory contains
        # all runs for the swiped standard deviation range and the evaluated bits.
        # RE to search for the "bits<number>" substring
        find_bits_re = re.compile(r'bits\d')
        # Get a unique set of all "bits<number>" available in all executed experiments
        bits_experimented = {extract_substring(find_bits_re, str(dir)) for dir in all_dirs}
        # Transform the set into a sorted list
        bits_experimented = natsort.humansorted(bits_experimented)

        # Get all paths to experiments executed for each bit
        bit_dir_grouped = [list(experiment_path.glob(f'./*{bit}*d{dim}')) for bit in bits_experimented]

        # Ensure all experiments executed for each bit are sorted so that dev 0.5
        # comes before dev 0.6, for example. The variable below is a list of lists. The
        # first dimension is equal to the number of shapes and the second is a
        # list of paths to each standard deviation experimented.
        bit_dir_grouped = [natsort.humansorted(i) for i in bit_dir_grouped]

        # This variable is a list of lists of numpy arrays. [<bits>, <std dev>, <seed>].
        bit_accs = [list(map(parse_accuracy_directory, dev_range)) for dev_range in bit_dir_grouped]

        # Transform bit_accs to a list of 2D numpy arrays for easier manipulation
        bit_accs = [np.vstack(deviations) for deviations in bit_accs]

        # Parse reference
        reference_acc = parse_accuracy_directory(reference_path)
        # Reshape reference_acc if it has been executed with more seeds
        reference_acc = reference_acc[0:bit_accs[0].shape[1]]

        # The data to be plotted is the accuacy difference to the reference
        data = [np.subtract(reference_acc, acc) for acc in bit_accs]

        ## Get the mean loss inflicted by sign quantization
        signquantize_accs = parse_accuracy_directory(sq_path)
        signquantize_accs = np.array(signquantize_accs)

        signquantize_loss = reference_acc - signquantize_accs
        sq_loss_mean = np.mean(signquantize_loss)

        return data, bits_experimented, sq_loss_mean

    def _create_paths(apps):
        experiment_paths = []
        reference_model_paths = []
        signquantize_paths = []
        for app in apps:
            experiment_paths.append(Path(f'_transformation/{app}/hdc/deviation'))
            signquantize_paths.append(Path(f'_transformation/{app}/hdc/signquantize/amsq-d{dim}'))
            reference_model_paths.append(Path(f'_accuracy/{app}/hdc/map-encf32-amf32-d{dim}'))

        return experiment_paths, reference_model_paths, signquantize_paths

    def _parse_paths(experiment_paths, reference_model_paths, signquantize_paths):
        # Parse data to be plotted. Data is a list of list of numpy arrays.
        # dim 0: Applications evaluated
        # dim 1: Bit experiments executed in this application
        # dim 2: A 2D numpy tensor in the shape [std_deviation, seed]
        data = []
        sq_losses = []
        labels = None
        for exp_path, ref_path, sq_path in zip(experiment_paths, reference_model_paths, signquantize_paths):
            accs, labels, sq_loss = _parse_app_results(exp_path, ref_path, sq_path, dim)
            data.append(accs)
            sq_losses.append(sq_loss)

        return data, sq_losses, labels

    def _get_data_from_apps(apps):
        ret = _create_paths(apps)
        data, sq_losses, labels = _parse_paths(*ret)
        return data, sq_losses, labels

    apps = ['voicehd', 'mnist', 'language', 'hdchog-fashionmnist']
    data, sq_losses, labels = _get_data_from_apps(apps)
    # Make the benchmark name as they appear in the plot
    apps = ['voicehd', 'mnist', 'language', 'hdchog']

    # Handle the EMG dataset since it has different subjects
    app = 'emg'
    apps.append(app)
    experiment_paths = []
    signquantize_paths = []
    reference_model_paths = []
    experiment_paths.append(Path(f'_transformation/{app}/hdc/all/deviation'))
    signquantize_paths.append(Path(f'_transformation/{app}/hdc/all/signquantize/amsq-d{dim}'))
    reference_model_paths.append(Path(f'_accuracy/{app}/hdc/all/map-encf32-amf32-d{dim}'))
    emg_data, emg_sq_losses, _ = _parse_paths(experiment_paths, reference_model_paths, signquantize_paths)
    data += emg_data
    sq_losses += emg_sq_losses

    # Handle GraphHD datasets also
    apps.append('graphhd')
    graphhd_data, graphhd_sq_losses, _ = _get_data_from_apps(GRAPHHD_DATASETS)
    graphhd_data = np.array(graphhd_data)
    # Get the mean of all datasets
    graphhd_data = np.mean(graphhd_data, axis=0)
    graphhd_sq_losses = np.mean(graphhd_sq_losses)

    data.append(graphhd_data)
    sq_losses.append(graphhd_sq_losses)

    # Transform to array
    data = np.array(data) # shape: [app, bit, deviation, seed]
    sq_losses = np.array(sq_losses) # shape: [apps]

    # Collapse the last dimension (seed) by computing its mean.
    data = np.mean(data, axis=-1)

    # Plot #
    # Create plotted range
    start = 0.1
    stop = 2.0
    step = 0.1
    dev_range = np.arange(start, stop+step, step)

    # Use latex notation on legend. The legend must be something like "$B1$".
    replace_bits = partial(str_replace, pattern='bits', to='B')
    labels = map(replace_bits, labels)
    labels = list(map(make_latex_equation, labels))

    # Allow global function to select only a subset of the experimented bits in
    # TQHD.
    data, labels = select_bits(data, labels)
    path = '_plots/error_deviation'
    if suplementary:
        path += f'-suplement-d{dim}'

    plot_deviation(
        data,
        dev_range,
        apps,
        labels=labels,
        hline=sq_losses,
        path=path+'.pdf'
    )
    plot_deviation(
        data,
        dev_range,
        apps,
        labels=labels,
        hline=sq_losses,
        path=path+'.png'
    )

    def _find_max_diff(data, app_names: List[str]):
        '''
        In the paper, we claim that choose P=[-1, 1] is a good choice since it
        provides good accuracy compared to the best possible. This function
        computes the difference between choosing the P limits equal to the
        standard deviation and the best choice.
        '''
        # The 1.0 result is the 9th index in the deviation dimension since it
        # goes from [0.1, 2.0]
        ind_1 = 9

        # In the paper, we claim that B2 suffers the biggest accuracy
        # fluctuation when choosing P. Discard all other bits and delete the
        # dimension. New shape [app][deviation]
        b2_data = data[:, 0, :]

        # Get the acc loss when P=\sigma
        claim_data = b2_data[:, ind_1]
        best_data = np.min(b2_data, axis=-1)

        print('--- Plot deviation claim ---')
        print('In the paper, we claim that choosing P=[-1std, 1std] is a good '
              'choice. Printing the difference between choosing 1std and the '
              'best possible for B=2.')
        print('P = 1std:')
        print_labeled(claim_data, app_names)
        print('Best P:')
        print_labeled(best_data, app_names)
        print('Difference:')
        print_labeled(best_data-claim_data, app_names)

    def _find_p1_progression(data, app_names: List[str]):
        # The 1.0 result is the 9th index in the deviation dimension since it
        # goes from [0.1, 2.0]
        ind_1 = 9

        # In the paper, we claim that B4> does not change accuracy
        # significantly for P=1 sigma. Discard all other values of P. New shape
        # [app][bits]
        all_bits = data[:, :, ind_1]

        # Compute accuracy loss drop when increasing bits B
        diff_bits = -np.diff(all_bits, axis=-1)

        # Get the acc loss when P=\sigma
        print('--- Plot deviation claim ---')
        print('In the paper, we claim that choosing B>4 result in diminishing '
              'gains. Each entry represents the accuracy difference of '
              'B_{i+1}-B{i}. For instance, the first entry for any application '
              'is B3-B2, and so on.')
        print_labeled(diff_bits, app_names)

    # Find biggest mean between best choice of P and 1.0
    if not suplementary:
        _find_max_diff(data, APP_PLOT_NAMES)
        _find_p1_progression(data, APP_PLOT_NAMES)

def plot_dimension(data, dim_range, apps, labels, sq_loss, **kwargs):
    """docstring for plot_dimension"""
    cm = 1/2.54  # centimeters in inches
    IEEE_column_width = 8.89*cm # Column width in IEEE paper in cms
    fig, axs = plt.subplots(
            nrows=2,
            ncols=2,
            #sharex=True,
            figsize=(IEEE_column_width*2.5, IEEE_column_width*2)
            )

    # Handle plot when sq_loss is not given
    if sq_loss is None:
        sq_loss = [None] * len(data)

    for app_data, ax, app_name, sq in zip(data, axs.flatten(), apps, sq_loss):
        min_y = float('inf')
        for curve, label in zip(app_data, labels):
            mean = np.mean(curve, axis=1)
            ax.plot(
                dim_range,
                mean,
                label=label
            )

            # This step is used to align the y axis to 0. This prevents
            # matplotlib to create y axis with different starting points.
            min = np.min(mean)
            min_y = min if min < min_y else min
            # TODO: This is a workaround to make the legend a little bit above
            # the plots since matplotlib tight_layout is not working correctly
            # for this plot.
            ax.set_title(' ')

        # Place application name at the right of the plot
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(
            app_name,
            rotation=270,
            rotation_mode='anchor',
            verticalalignment="bottom",
            **APP_NAME_STYLE,
        )

        if sq is not None:
            mean = np.mean(sq, axis=1)
            ax.plot(
                dim_range,
                mean,
                label='$SQ$',
                color='black',
                linestyle='--',
            )

        ax.autoscale(enable=True, axis='x', tight=True)
        # Check if any plotted data is below 0
        if min_y > 0:
            ax.set_ylim(ymin=np.floor(min_y))

        # Adjust xticks to be in multiples of 0.2 and to end exactly at the end of
        # dev_range parameter.
        #ax.set_xticks(xticks_pos)
        #ax.set_xticklabels(xticks_str)

    handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, ncols=len(labels), loc='outside upper center', bbox_to_anchor=(1,1))
    #fig.legend(handles, labels, ncols=len(labels), bbox_to_anchor=(1,1.005))
    fig.legend(handles, labels, loc='outside upper center', ncols=len(data[0])+1, fontsize='x-small')

    fig.supylabel('Accuracy Loss in pp')
    fig.supxlabel(r'Dimensions')

    plt.tight_layout()
    plot._plot(**kwargs)

def parse_compaction_app(app_dir: str, dim_range, configs):
    def _parse_csv(path: Path):
        data = pd.read_csv(path)
        return data

    def _parse_compaction_dir(p: Path):
        data = map_sorted_subfiles(p, _parse_csv)
        df = pd.concat(data)
        return df

    def _parse_compresion_bits_dir(p: Path):
        datasets = list(map_sorted_subfolders(p, _parse_compaction_dir))
        dataset = pd.concat(datasets) # Return a single dataframe

        return dataset

    def _parse_dim_dir(p: Path):
        datasets = list(map_sorted_subfolders(p, _parse_compresion_bits_dir))
        dataset = pd.concat(datasets) # Return a single dataframe

        return dataset

    def _parse_tqhd_bit_dir(p: Path):
        datasets = list(map_sorted_subfolders(p, _parse_dim_dir))
        dataset = pd.concat(datasets)

        # TODO: Workaround to add execution type information to the dataframe
        # since compaction.py does not export the type of encoding used and
        # whether the RLE was interleaved.
        execution_type = p.stem
        dataset.insert(0, "execution_type", execution_type)
        return dataset

    def _parse_all(p: str):
        path = Path(p)
        datasets = list(map_sorted_subfolders(path, _parse_tqhd_bit_dir))
        dataset = pd.concat(datasets)

        return dataset

    df = _parse_all(app_dir)

    # New version #
    # Use information in CSV
    def _calc_compacted_size_improvement(df, keys, dimensions):
        compaction_bits = int(keys[1])
        query_df = df.query(f'execution_type == "{keys[0]}" & tqhd_b == {keys[1]} & compaction_bits_0 == {keys[2]} & compaction_bits_1 == {keys[3]}')

        tqhd_am_sizes = query_df['tqhd_am_size'].to_numpy()
        comp_am_sizes = query_df['comp_am_size'].to_numpy()
        improvement = np.mean(comp_am_sizes/tqhd_am_sizes)

        return improvement

    # Create partial function with closure
    calc_compacted_improvement = lambda keys: _calc_compacted_size_improvement(df, keys, dim_range)
    compacted_sizes = list(map(calc_compacted_improvement, configs))
    compacted_improvement = np.array(compacted_sizes)

    return 1-compacted_improvement

def figure_compaction():
    # Create plotted range
    start = 1000
    stop = 10000
    step = 1000
    dim_range = np.arange(start, stop+step, step)

    configs = [
            # <bits> <compaction-bits>
            ['BaseZero-interleave', '3', '3', '3'],
            ['BaseZero-interleave', '4', '3', '3'],
            ['BaseZero-interleave', '4', '4', '4'],
            ['BaseZero-interleave', '5', '4', '4'],
            ['BaseZero-interleave', '6', '4', '4'],
            ['BaseZero-interleave', '7', '4', '4'],
            ['BaseZero-interleave', '8', '4', '4'],
        ]

    def _parse_compaction_app(app_name: str):
        path = f'_transformation/{app_name}/hdc/compaction'
        compaction_rate = parse_compaction_app(path, dim_range, configs)
        return np.array(compaction_rate)

    def _parse_multidataset_app(app_names: List[str]):
        compaction_rates = list(map(_parse_compaction_app, app_names))
        compaction_rates = np.mean(compaction_rates, axis=0)
        return np.array(compaction_rates)

    app_names = ['voicehd', 'mnist', 'language', 'hdchog', 'emg', 'graphhd']

    apps = ['voicehd', 'mnist', 'language', 'hdchog-fashionmnist']
    compaction_rates = list(map(_parse_compaction_app, apps))
    compaction_rates = np.array(compaction_rates)

    emg_apps = ['emg-s0', 'emg-s1', 'emg-s2', 'emg-s3', 'emg-s4']
    emg_rates = _parse_multidataset_app(emg_apps)
    compaction_rates = np.vstack((compaction_rates, emg_rates))

    global GRAPHHD_DATASETS
    graphhd_apps = GRAPHHD_DATASETS
    graphhd_rates = _parse_multidataset_app(graphhd_apps)
    compaction_rates = np.vstack((compaction_rates, graphhd_rates))

    compaction_rates *= 100 # Make percentage
    # Add mean column at the end of array
    means = np.mean(compaction_rates, axis=0).reshape(1, -1)
    table = np.transpose(np.vstack((compaction_rates, means)))

    headers = [*app_names, 'mean']
    config_labels = [f'{execution_type}-$B{tqhd_bits}C_0{c0_size}C_1{c1_size}$' for execution_type, tqhd_bits, c0_size, c1_size in configs]
    configs_df = pd.DataFrame({'config': config_labels})

    table_df = pd.DataFrame(data=table, columns=headers)

    table_df = pd.concat([configs_df, table_df], axis=1)

    latex_alignment = 'c' * table_df.columns.size
    float_format = lambda s: '{:.1f}\%'.format(s)
    latex = table_df.to_latex(
            index=False, # No row number
            float_format=float_format,
            column_format=latex_alignment,
            caption='Memory footprint reduction in compacted AM.',
            label='tab:compaction'
            )
    print(latex)
    with open('_plots/compaction.tex', 'w') as f:
        print(latex, file=f)

def plot_tqhd_vs_pqhdc(
        data_tqhd,
        data_pqhdc,
        dims,
        apps,
        labels,
        colors=None,
        tqhd_marker='o',
        tqhd_linestyle='-',
        pqhdc_marker='s',
        pqhdc_linestyle=':',
        xlabel='Dimensions',
        xaxis_tick_formatter=None,
        **kwargs):
    """docstring for plot_tqhd_vs_pqhdc"""
    ax_size = len(data_tqhd)
    cm = 1/2.54  # centimeters in inches
    IEEE_column_width = 8.89*cm # Column width in IEEE paper in cms
    fig, axs = plt.subplots(
            nrows=3,
            ncols=2,
            #sharex=True,
            figsize=(IEEE_column_width*2.5, IEEE_column_width*2)
            )

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    for app_tqhd, app_pqhdc, ax, app_name in zip(data_tqhd, data_pqhdc, axs.flatten(), apps):
        min_y = float('inf')
        #for curve, label in zip(app_data, labels):
        for curve_tqhd, curve_pqhdc, label, color in zip(app_tqhd, app_pqhdc, labels, colors):
            ax.plot(
                dims,
                curve_tqhd,
                label=label,
                linestyle=tqhd_linestyle,
                marker=tqhd_marker, # Circle marker
                color=color
            )

            ax.plot(
                dims,
                curve_pqhdc,
                label=label,
                linestyle=pqhdc_linestyle,
                color=color, # Plot both lines with the same color
                marker=pqhdc_marker, # Square marker
            )

            if xaxis_tick_formatter:
                ax.xaxis.set_major_formatter(xaxis_tick_formatter)

            ## This step is used to align the y axis to 0. This prevents
            ## matplotlib to create y axis with different starting points.
            #min = np.min(mean)
            #min_y = min if min < min_y else min
            ## TODO: This is a workaround to make the legend a little bit above
            ## the plots since matplotlib tight_layout is not working correctly
            ## for this plot.
            ax.set_title(' ')

        # Place application name at the right of the plot
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(
            app_name,
            rotation=270,
            rotation_mode='anchor',
            verticalalignment="bottom",
            **APP_NAME_STYLE,
        )

        #if sq is not None:
        #    mean = np.mean(sq, axis=1)
        #    ax.plot(
        #        dim_range,
        #        mean,
        #        label='$SQ$',
        #        color='black',
        #        linestyle='--',
        #    )

        ax.autoscale(enable=True, axis='x', tight=True)
        ## Check if any plotted data is below 0
        #if min_y > 0:
        #    ax.set_ylim(ymin=np.floor(min_y))

        # Adjust xticks to be in multiples of 0.2 and to end exactly at the end of
        # dev_range parameter.
        #ax.set_xticks(xticks_pos)
        #ax.set_xticklabels(xticks_str)

    # Create legend handles
    data = data_tqhd[0]
    patches = []
    for _, color in zip(data, colors):
        patch = mpatches.Patch(color=color, label='The red data')
        patches.append(patch)
    #axs.flatten().plot([],[], color='black', marker=tqhd_marker)
    patch = mlines.Line2D([], [], linestyle=tqhd_linestyle, color='black', marker=tqhd_marker)
    patches.append(patch)
    patch = mlines.Line2D([], [], linestyle=pqhdc_linestyle, color='black', marker=pqhdc_marker)
    patches.append(patch)
    legend_labels = labels+['TQHD', 'PQ-HDC']

    #handles, labels = axs.flatten()[0].get_legend_handles_labels()
    #fig.legend(handles, labels, ncols=len(labels), loc='outside upper center', bbox_to_anchor=(1,1))
    #fig.legend(handles, labels, ncols=len(labels), bbox_to_anchor=(1,1.005))
    fig.legend(patches, legend_labels, loc='outside upper center', ncols=len(data_tqhd[0])+2, fontsize='small')

    fig.supylabel('Accuracy Loss in pp')
    fig.supxlabel(xlabel)

    plt.tight_layout()
    plot._plot(**kwargs)

def figure_tqhd_vs_pqhdc():
    """docstring for figure_tqhd_vs_pqhdc"""
    apps = ['voicehd', 'mnist', 'language']
    def _parse_dim_dir(path: str):
        accs = parse_accuracy_directory(path)
        return accs

    def _parse_bit_dir(path: str):
        dims = map_sorted_subfolders(path, _parse_dim_dir)
        dims = list(dims)
        return dims

    def _parse_app_dir(path: str):
        bits = list(map_sorted_subfolders(path, _parse_bit_dir))
        bits = np.array(bits)
        return bits

    def _parse_app(acc_path: str, ref_path: str):
        quantized_accs = _parse_app_dir(acc_path)
        reference_accs = parse_app_reference_dir(ref_path)

        acc_loss = reference_accs - quantized_accs

        return acc_loss

    # Parse TQHD
    losses_tqhd = []
    for app in apps:
        acc_path = f'_transformation/{app}/hdc/paper-tqhd'
        ref_path = f'_accuracy/{app}/hdc'
        loss = _parse_app(acc_path, ref_path)
        losses_tqhd.append(loss)

    app = 'emg'
    loss = _parse_app(
            f'_transformation/{app}/hdc/all/paper-tqhd',
            f'_accuracy/{app}/hdc/all'
            )
    losses_tqhd.append(loss)

    # Parse PQHDC
    acc_paths = []
    ref_paths = []
    for app in apps:
        acc_path = f'_transformation/{app}/hdc/pqhdc'
        ref_path = f'_accuracy/{app}/hdc'
        acc_paths.append(acc_path)
        ref_paths.append(ref_path)

    acc_paths.append(f'_transformation/emg/hdc/all/pqhdc')
    ref_paths.append(f'_accuracy/emg/hdc/all')

    losses_pqhdc = []
    for acc_path, ref_path in zip(acc_paths, ref_paths):
        loss = _parse_app(acc_path, ref_path)
        losses_pqhdc.append(loss)

    # Remove results for B3, and B4 for TQHD and PQHDC since they B2 can
    # already quatize EMG pretty well.
    #losses_tqhd[-1] = np.expand_dims(losses_tqhd[-1][0], axis=0)
    #losses_pqhdc[-1] = np.expand_dims(losses_pqhdc[-1][0], axis=0)

    #losses_pqhdc = np.array(losses_pqhdc)
    dim_start = 1000
    dim_end = 10000
    dim_step = 1000
    dims = np.arange(dim_start, dim_end+dim_step, dim_step)
    labels = ['$B2$', '$B3$', '$B4$']
    apps += ['emg']
    # Filter bits

    for i in range(len([*zip(losses_tqhd, losses_pqhdc)])):
        losses_tqhd[i] = losses_tqhd[i][0:3]
        losses_pqhdc[i] = losses_pqhdc[i][0:3]
    plot_tqhd_vs_pqhdc(losses_tqhd, losses_pqhdc, dims, apps, labels=labels, path='_plots/tqhd_vs_pqhdc.pdf')
    plot_tqhd_vs_pqhdc(losses_tqhd, losses_pqhdc, dims, apps, labels=labels, path='_plots/tqhd_vs_pqhdc.png')

    def _scalability(data_tqhd, data_pqhdc):
        '''
        In the paper, we want to show that TQHD provides better accuracy when
        given more resources and to showcase its potential in difficult
        scenarios, i.e., low dimensions and voicehd/mnist.
        '''
        # Ignore language and emg. Create arrays in the shape:
        # [app][bit][dimensions][seed]
        tqhd = np.array(data_tqhd[0:2])
        pqhdc = np.array(data_pqhdc[0:2])

        # Get the mean of all seeds and collapse [seed] dimension
        mean_tqhd = np.mean(tqhd, axis=-1)
        mean_pqhdc = np.mean(pqhdc, axis=-1)

        # Get results only for D=1000 and remove [dimensions]
        dim_tqhd = mean_tqhd[..., 0]
        dim_pqhdc = mean_pqhdc[..., 0]

        # Print acc improvement when increasing bits
        improvement_tqhd = np.diff(dim_tqhd, axis=-1)
        improvement_pqhdc = np.diff(dim_pqhdc, axis=-1)

        print('TQHD B2 D=1000:')
        print(dim_tqhd)
        print('Improvement TQHD:')
        print(improvement_tqhd)
        print('PQ-HDC B2 D=1000:')
        print(dim_pqhdc)
        print('Improvement PQ-HDC:')
        print(improvement_pqhdc)
    _scalability(losses_tqhd, losses_pqhdc)

def plot_accuracy(
        x,
        data,
        apps,
        colors=None,
        xlabel='Dimensions',
        legend_dict=None,
        **kwargs):
    """docstring for plot_tqhd_vs_pqhdc"""
    cm = 1/2.54  # centimeters in inches
    IEEE_column_width = 8.89*cm # Column width in IEEE paper in cms
    cols = 2
    rows = math.ceil(data[0]['y'].shape[0]/cols)
    fig, axs = plt.subplots(
            nrows=rows,
            ncols=cols,
            #sharex=True,
            figsize=(IEEE_column_width*2.5, IEEE_column_width*2)
            )

    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    def _plot_technique(technique, axs):
        apps = technique['y']
        linestyle = technique['linestyle']
        marker = technique['marker']
        for app, ax in zip(apps, axs.flatten()):
            for curve, color in zip(app, colors):
                # Collapse the last dimension (seeds) and obtain the mean of
                # all seeds for each dimension entry.
                ax.plot(
                    x,
                    curve,
                    linestyle=linestyle,
                    marker=marker,
                    color=color
                )

    for d in data:
        _plot_technique(d, axs)

    #for app_tqhd, app_pqhdc, ax, app_name in zip(data_tqhd, data_pqhdc, axs.flatten(), apps):
    #    min_y = float('inf')
    #    #for curve, label in zip(app_data, labels):
    #    quanthd_marker = '^'
    #    quanthd_linestyle = '-'
    #    markers = cycler(
    #            marker=[tqhd_marker, pqhdc_marker, quanthd_marker],
    #            linestyle=[tqhd_linestyle, pqhdc_linestyle, quanthd_linestyle]
    #            )
    #    for curve_tqhd, curve_pqhdc, label, color in zip(app_tqhd, app_pqhdc, labels, colors):

    #        ## This step is used to align the y axis to 0. This prevents
    #        ## matplotlib to create y axis with different starting points.
    #        #min = np.min(mean)
    #        #min_y = min if min < min_y else min
    #        ## TODO: This is a workaround to make the legend a little bit above
    #        ## the plots since matplotlib tight_layout is not working correctly
    #        ## for this plot.
    #        ax.set_title(' ')

    # Place application name at the right of each plot
    for ax, app_name in zip(axs.flatten(), apps):
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(
            app_name,
            rotation=270,
            rotation_mode='anchor',
            verticalalignment="bottom",
            **APP_NAME_STYLE,
        )

    #    ax.autoscale(enable=True, axis='x', tight=True)
        ## Check if any plotted data is below 0
        #if min_y > 0:
        #    ax.set_ylim(ymin=np.floor(min_y))

    # TODO: This is a workaround to make the legend a little bit above
    # the plots since matplotlib tight_layout is not working correctly
    # for this plot.
    for ax in axs.flatten():
        ax.set_title(' ')
    if legend_dict:
        fig.legend(**legend_dict, loc='outside upper center', fontsize='small')

    fig.supylabel('Accuracy Loss in pp')
    fig.supxlabel(xlabel)

    plt.tight_layout()
    plot._plot(**kwargs)

def plot_tqhd_vs_all_bar(
        data_tqhd,
        data_pqhdc,
        data_quanthd,
        dims,
        apps,
        labels,
        colors=None,
        tqhd_hatch='o',
        pqhdc_hatch='X',
        quanthd_hatch='*',
        xlabel='Dimensions',
        xaxis_tick_formatter=None,
        xticks_labels="",
        **kwargs):
    """docstring for plot_tqhd_vs_pqhdc"""
    ax_size = len(data_tqhd)
    cm = 1/2.54  # centimeters in inches
    IEEE_column_width = 8.89*cm # Column width in IEEE paper in cms
    fig, axs = plt.subplots(
            nrows=5,
            ncols=1,
            #sharex=True,
            figsize=(IEEE_column_width*4, IEEE_column_width*2)
            )

    #data_tqhd = np.mean(data_tqhd, axis=-1)
    #data_pqhdc = np.mean(data_pqhdc, axis=-1)
    #data_quanthd = np.mean(data_quanthd, axis=-1)
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    hatches = [pqhdc_hatch, quanthd_hatch, tqhd_hatch]

    num_dims = data_tqhd.shape[-2]
    x = np.arange(num_dims)
    width = 0.09
    multiplier = 0
    for app_tqhd, app_pqhdc, app_quanthd, ax, app_name in zip(data_tqhd, data_pqhdc, data_quanthd, axs.flatten(), apps):
        multiplier = 0
        for technique, hatch in zip([app_pqhdc, app_quanthd, app_tqhd], hatches):
            # Get the mean of all seeds
            mean = np.mean(technique, axis=-1)
            for scenario, color in zip(mean, colors):
                offset = width*multiplier
                rects = ax.bar(x+offset, scenario, width, hatch=hatch, edgecolor='black', color=color)
                multiplier += 1
            # Adjust xticks to the middle of each bar group
            ax.set_xticks(x + (width*(multiplier-1)/2), xticks_labels, fontstyle='italic')

        # Place application name at the right of the plot
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(
            app_name,
            rotation=270,
            rotation_mode='anchor',
            verticalalignment="bottom",
            **APP_NAME_STYLE,
        )

    # Create legend handles
    data = data_tqhd[0]
    patches = []
    for _, color in zip(data, colors):
        patch = mpatches.Patch(facecolor=color, label='The red data', edgecolor='black')
        patches.append(patch)

    patch = mpatches.Patch(facecolor='white', hatch=pqhdc_hatch, edgecolor='black')
    patches.append(patch)
    patch = mpatches.Patch(facecolor='white', hatch=quanthd_hatch, edgecolor='black')
    patches.append(patch)
    patch = mpatches.Patch(facecolor='white', hatch=tqhd_hatch, edgecolor='black')
    patches.append(patch)
    legend_labels = labels+['PQ-HDC', 'QuantHD', 'TQHD']

    #handles, labels = axs.flatten()[0].get_legend_handles_labels()
    #fig.legend(patches, legend_labels, loc='outside upper center', ncols=len(data_tqhd[0])+2, fontsize='small')
    fig.legend(patches, legend_labels, loc='outside upper center', ncols=9, fontsize='small')

    # TODO: This is a workaround to make the legend a little bit above the
    # plots since matplotlib tight_layout is not working correctly for this
    # plot.
    axs.flatten()[0].set_title(' ')

    fig.supylabel('Accuracy Loss in pp')
    #fig.supxlabel(xlabel)

    plt.tight_layout()
    plot._plot(**kwargs)

def figure_tqhd_vs_all():
    """docstring for figure_tqhd_vs_pqhdc"""
    apps = [
            'voicehd',
            'mnist',
            'language',
            'hdchog-fashionmnist',
        ]
    acc_ref = get_reference_accs()

    def _parse_transformation(transformation_name: str, emg_all: bool=True) -> NDArray:
        acc_apps = parse_transformation_apps(apps, transformation_name)
        acc_emg = parse_transformation_emg(transformation_name, all=emg_all)
        acc_graphhd_all = parse_transformation_apps(GRAPHHD_DATASETS, transformation_name)
        acc_graphhd = np.mean(acc_graphhd_all, axis=0)

        # TODO: This can be removed later
        # Make sure all apps for the same transformation have the same number of
        # bits/retraining iterations
        lenghts = map(len, acc_apps)
        min_len = min(lenghts)
        min_len = min(min_len, len(acc_emg), len(acc_graphhd))
        acc_apps = [acc[:min_len] for acc in acc_apps]
        acc_emg = acc_emg[:min_len]
        acc_graphhd = acc_graphhd[:min_len]

        acc_all = np.concatenate((acc_apps, [acc_emg], [acc_graphhd]))

        return acc_all

    def _parse_transformation_technique(technique_name: str, acc_ref: NDArray, emg_all: bool) -> NDArray:
        # acc.shape = [app, bit/retraining, dim, seed]
        acc = _parse_transformation(technique_name, emg_all=emg_all)

        # Swap axes to compute losses #
        # Make acc.shape compatible with acc_ref shape, which is [app, dim, seed]
        # acc.shape = [bit/retraining, app, dim, seed]
        acc = np.swapaxes(acc, 0, 1)
        losses = acc_ref - acc
        # Bring apps to the first dimension as:
        # losses.shape = [app, bit/retraining, dim, seed]
        losses = np.swapaxes(losses, 0, 1)

        # Collapse last dimension of losses by computing the mean of all seeds
        losses = np.mean(losses, axis=-1)

        # Collapse the last dimension
        return losses

    losses_tqhd = _parse_transformation_technique('paper-tqhd', acc_ref, emg_all=True)
    losses_pqhdc = _parse_transformation_technique('pqhdc', acc_ref, emg_all=True)
    losses_quanthdbin = _parse_transformation_technique('quanthdbin', acc_ref, emg_all=False)

    # Optional: Remove results for B3, and B4 for TQHD and PQHDC since they B2
    # can already quantize EMG pretty well.
    #losses_tqhd[-1] = np.expand_dims(losses_tqhd[-1][0], axis=0)
    #losses_pqhdc[-1] = np.expand_dims(losses_pqhdc[-1][0], axis=0)

    #losses_pqhdc = np.array(losses_pqhdc)
    dim_start = 1000
    dim_end = 10000
    dim_step = 1000
    dims = np.arange(dim_start, dim_end+dim_step, dim_step)

    # Filter bits #
    # This plot shows only bits 2 to 4 and not the entire experiment space.
    # Make sure all quantization results arrays contain the same amount of
    # bits.
    losses_tqhd = losses_tqhd[:, 0:3]
    losses_pqhdc = losses_pqhdc[:, 0:3]

    # Retrieve the a subset of retraining results of QuantHD #
    # Even though we train QuantHD for several iterations, the plot depicts
    # only small subset of the iterations as more epochs provide accuracy
    # better than the reference models since the reference models do not have
    # retraining.
    # Retrieve 3rd, 6th, and 9th retraining epoch.
    losses_quanthdbin = losses_quanthdbin[:,2:5:1]

    # Create the labels below the xticks
    dimensions_str = ['D'+str(dim) for dim in np.arange(1000, 11000, 1000)]
    colors1 = None
    colors2 = [
        '#97ca8eff',
        '#8889c7ff',
        '#cccb90ff'
    ]
    colors3 = [
        '#6dd499ff',
        '#9ecbedff',
        '#fe9d52ff'
            ]
    # Pack data to pass to the plot function #
    apps += ['emg']
    apps += ['graphhd']
    data = np.array([
        losses_tqhd,
        losses_pqhdc,
        losses_quanthdbin
        ]
    )
    tqhd = {
        'y': losses_tqhd,
        'marker': 'o',
        'linestyle': '-',
        'label': 'TQHD',
    }

    pqhdc = {
        'y': losses_pqhdc,
        'marker': 's',
        'linestyle': ':',
        'label': 'PQ-HDC',
    }

    quanthdbin = {
        'y': losses_quanthdbin,
        'marker': '^',
        'linestyle': ':',
        'label': 'QuantHD',
    }

    data = [
            tqhd,
            pqhdc,
        ]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    labels = ['$B2$', '$B3$', '$B4$']

    # Create Custom Legend #
    # Create a custom figure legend with matplotlib patches
    patches = []
    for _, color in zip(labels, colors):
        patch = mpatches.Patch(color=color, label='The red data')
        patches.append(patch)

    markers = [d['marker'] for d in data]
    linestyles = [d['linestyle'] for d in data]
    for marker, linestyle in zip(markers, linestyles):
        patch = mlines.Line2D([], [], linestyle=linestyle, color='black', marker=marker)
        patches.append(patch)

    technique_names = [d['label'] for d in data]
    legend_labels = labels+technique_names
    # TODO: Tornar a criação da legenda flexível. Isso vai permitir chamar o plot com variações diferentes de "data".

    legend_dict = {
            'handles': patches,
            'labels': legend_labels,
            'ncols': len(legend_labels)
            }

    # Make TQHD vs PQHDC
    plot_accuracy(dims, data, APP_PLOT_NAMES, path='_plots/tqhd_vs_pqhdc.pdf', colors=colors, legend_dict=legend_dict, xlabel='Dimensions')

    # TQHD vs QuantHD #
    patches = []
    legend_labels = []
    data = [tqhd, quanthdbin]
    tqhd_labels = ['$B2$', '$B3$', '$B4$']
    quanthdbin_labels = ['$R2$', '$R3$', '$R4$']

    # Add a legend entry for each technique
    for d in data:
        technique_name = d['label']
        linestyle = d['linestyle']
        marker = d['marker']
        patches.append(
            mlines.Line2D([], [], linestyle=linestyle, color='black', marker=marker)
        )
        legend_labels.append(technique_name)

    # Add legend entry to each type of line
    def _handles_tqhd_vs_quanthd(data_dict, labels):
        marker = data_dict['marker']
        linestyle = data_dict['linestyle']
        patches = []
        for _, color in zip(labels, colors):
            patch = mlines.Line2D([], [], linestyle=linestyle, color=color, marker=marker)
            patches.append(patch)
        return patches
    patches += _handles_tqhd_vs_quanthd(tqhd, tqhd_labels)
    patches += _handles_tqhd_vs_quanthd(quanthdbin, quanthdbin_labels)
    legend_labels += tqhd_labels + quanthdbin_labels

    legend_dict = {
            'handles': patches,
            'labels': legend_labels,
            'ncols': len(legend_labels)
        }
    # Make TQHD vs QuantHD
    #plot_accuracy(dims, data, APP_PLOT_NAMES, path='_plots/tqhd_vs_quanthd.pdf', colors=colors, legend_dict=legend_dict, xlabel='Dimensions')

    # TODO: This will need adjustments for the paper discussion
    def _scalability(data_tqhd, data_pqhdc, data_quanthd):
        '''
        In the paper, we want to show that TQHD provides better accuracy when
        given more resources and to showcase its potential in difficult
        scenarios, i.e., low dimensions and voicehd/mnist.
        '''
        tqhd = data_tqhd
        pqhdc = data_pqhdc
        quanthd = data_quanthd

        # Get results only for D=1000 and remove [dimensions]
        dim_tqhd = tqhd[..., 0]
        dim_pqhdc = pqhdc[..., 0]
        dim_quanthd = quanthd[..., 0]

        # Print acc improvement when increasing bits
        improvement_tqhd = np.diff(dim_tqhd, axis=-1)
        improvement_pqhdc = np.diff(dim_pqhdc, axis=-1)
        improvement_quanthd = np.diff(dim_quanthd, axis=-1)

        print('TQHD D=1000:')
        print_labeled(dim_tqhd, APP_PLOT_NAMES)
        print('Improvement TQHD when increasing B for D=1000 (<0 values mean accuracy loss reduction):')
        print_labeled(improvement_tqhd, APP_PLOT_NAMES)
        print('PQ-HDC D=1000:')
        print_labeled(dim_pqhdc, APP_PLOT_NAMES)
        print('Improvement PQ-HDC when increasing B for D=1000 (<0 values mean accuracy loss reduction):')
        print_labeled(improvement_pqhdc, APP_PLOT_NAMES)
        print('QuantHD D=1000:')
        print_labeled(dim_quanthd, APP_PLOT_NAMES)
        print('Improvement QuantHD when increasing B for D=1000 (<0 values mean accuracy loss reduction):')
        print_labeled(improvement_quanthd, APP_PLOT_NAMES)
        print('TQHD (B2) - QuantHD (Rmax) for D=1K. <0 results indicate that TQHD is better.')
        print_labeled(dim_tqhd[:,0]-dim_quanthd[:,-1], APP_PLOT_NAMES)
    _scalability(losses_tqhd, losses_pqhdc, losses_quanthdbin)

def figure_noise():
    """docstring for figure_noise"""
    acc_ref = get_reference_accs()

    def _parse_dim_dir(path: str):
        accs = parse_accuracy_directory(path)
        return accs

    def _parse_bit_dir(path: str):
        dims = map_sorted_subfolders(path/'d1000', _parse_dim_dir)
        dims = list(dims)
        return dims

    def _parse_app_dir(path: str):
        bits = list(map_sorted_subfolders(path, _parse_bit_dir))
        bits = np.array(bits)
        return bits

    def _parse_app(acc_path: str):
        quantized_accs = _parse_app_dir(acc_path)

        return quantized_accs

    def _parse_transformation(transformation_name: str, emg_all: bool=True) -> NDArray:
        global APP_DIR_NAMES
        apps = APP_DIR_NAMES
        app_paths = [f'_transformation/{app}/hdc/{transformation_name}' for app in apps]
        acc_apps = list(map(_parse_app, app_paths))

        global GRAPHHD_DATASETS
        apps_graphhd = GRAPHHD_DATASETS
        graphhd_paths = [f'_transformation/{app}/hdc/{transformation_name}' for app in apps_graphhd]
        acc_graphhd_all = list(map(_parse_app, graphhd_paths))
        acc_graphhd = np.mean(acc_graphhd_all, axis=0)

        ## TODO: This can be removed later
        ## Make sure all apps for the same transformation have the same number of
        ## bits/retraining iterations
        #lenghts = map(len, acc_apps)
        #min_len = min(lenghts)
        #min_len = min(min_len, len(acc_graphhd))
        #acc_apps = [acc[:min_len] for acc in acc_apps]
        #acc_graphhd = acc_graphhd[:min_len]

        acc_all = np.concatenate((acc_apps, [acc_graphhd]))

        return acc_all

    def _parse_technique(technique_name: str, acc_ref: NDArray, emg_all: bool) -> NDArray:
        # acc.shape = [app, bit/retraining, dim, seed]
        acc = _parse_transformation(technique_name, emg_all=emg_all)

        # Swap axes to compute losses #
        # Make acc.shape compatible with acc_ref shape, which is [app, dim, seed]
        # acc.shape = [bit/retraining, app, dim, seed]
        acc = np.swapaxes(acc, 0, 1)
        losses = acc_ref - acc
        # Bring apps to the first dimension as:
        # losses.shape = [app, bit/retraining, dim, seed]
        losses = np.swapaxes(losses, 0, 1)

        # Collapse last dimension of losses by computing the mean of all seeds
        losses = np.mean(losses, axis=-1)

        # Collapse the last dimension
        return losses

    losses_tqhd = _parse_technique('paper-fault/tqhd', acc_ref, emg_all=True)
    losses_pqhdc = _parse_technique('paper-fault/pqhdc', acc_ref, emg_all=True)

    start = 1
    end = 10
    step = 1
    noise_percents = np.arange(start, end+step, step)
    labels = ['$B2$', '$B3$', '$B4$']
    apps = APP_PLOT_NAMES

    # Filter bits
    for i in range(len([*zip(losses_tqhd, losses_pqhdc)])):
        losses_tqhd[i] = losses_tqhd[i][0:3]
        losses_pqhdc[i] = losses_pqhdc[i][0:3]
    a = [losses_tqhd, losses_pqhdc, noise_percents, apps]
    kw = {
            'xlabel': 'Noise (%)',
            'labels': labels,
            'xaxis_tick_formatter': ticker.PercentFormatter(decimals=0),
            }
    plot_tqhd_vs_pqhdc(*a, **kw, path='_plots/noise.pdf')
    plot_tqhd_vs_pqhdc(*a, **kw, path='_plots/noise.png')

def main():
    #figure_normal_distribution()
    #figure_error_deviation()
    figure_compaction()
    #figure_tqhd_vs_all()
    #figure_noise()

    # Suplementary deviation experiment
    # This loops extends the design space exploration to also sweep dimensions
    # in D=[2K, 10K].
    #for i in range(2000, 10000+1, 1000):
    #    figure_error_deviation(i, suplementary=True)

if __name__ == '__main__':
    main()


