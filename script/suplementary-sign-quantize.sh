#!/bin/bash

# Quantize models using Sign Quantization
# This script expands the deviation experiment presented in the paper by
# including a broader number of dimensions D = [2000,10000] in steps of 1000.

set -eu

source script/common-sign-quantize.sh

# Overwrite DIMENSIONs parameter to consider the effects of deviation in a
# broader larger number of dimensions.
start=2000
step=1000
final=10000
DIMENSIONS=$(seq $start $step $final)

JOBS=12 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

enable_venv
cmd=""
cmd+=$(voicehd)
cmd+=$(emg)
cmd+=$(mnist)
cmd+=$(language)
cmd+=$(hdchog)
cmd+=$(com_graphhd)

#printf "$cmd"
parallel_launch "$JOBS" "$cmd"
disable_venv
