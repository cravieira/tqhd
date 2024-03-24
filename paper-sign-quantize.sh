#!/bin/bash

# Quantize models using Sign Quantization
# This script trains models with dfferent random seeds and then quantize them
# using Sign Quantization. The purpose of this experiment is to understand how
# the this naive quantization approach affects accuracy of the models.

set -eu

source script/common-sign-quantize.sh

JOBS=10 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

enable_venv
cmd=""
cmd+=$(voicehd)
cmd+=$(emg)
cmd+=$(mnist)
cmd+=$(language)
cmd+=$(graphhd)

#printf "$cmd"
parallel_launch "$JOBS" "$cmd"
disable_venv
