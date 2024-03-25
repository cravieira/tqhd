#!/bin/bash
#
# Download datasets used in the paper

source common.sh

enable_venv
py_launch "src/voicehd_hdc.py"
py_launch "src/mnist_hdc.py"
py_launch "src/language.py"
py_launch "src/emg.py"
py_launch "src/graphhd.py --dataset DD"
py_launch "src/hdchog.py --dataset FashionMNIST"
disable_venv
