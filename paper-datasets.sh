#!/bin/bash

# Download datasets used in the paper.
set -eu

source common.sh

function graphhd() {
    local dataset=$1
    py_launch "src/graphhd.py --dataset $dataset"
}

enable_venv
py_launch "src/voicehd_hdc.py"
py_launch "src/mnist_hdc.py"
py_launch "src/language.py"
py_launch "src/emg.py"
py_launch "src/graphhd.py --dataset DD"
com_foreach 'graphhd' 'GRAPHHD_DATASETS'
py_launch "src/hdchog.py --dataset FashionMNIST"
disable_venv
