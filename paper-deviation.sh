#!/bin/bash

# Quantize models using TQHD with different deviation values.
# This script executes the experiments as required for the paper. Thus, it
# fixes D=1000 and sweeps the P range from [-0.1sigma, 0.1sigma] to [-2.0, 2.0].

set -eu

source script/common-deviation.sh

JOBS=12 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

enable_venv
cmd=""
#cmd+=$(voicehd)
#cmd+=$(emg)
#cmd+=$(mnist)
#cmd+=$(language)
#cmd+=$(hdchog)
cmd+=$(com_graphhd)

#printf "$cmd"
parallel_launch "$JOBS" "$cmd"
disable_venv
