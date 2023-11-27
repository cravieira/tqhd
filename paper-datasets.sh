#!/bin/bash
#
# Download datasets used in the paper

source common.sh

py_launch "src/voicehd_hdc.py"
py_launch "src/mnist_hdc.py"
py_launch "src/language.py"
py_launch "src/emg.py"
