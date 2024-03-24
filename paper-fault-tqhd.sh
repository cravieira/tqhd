#!/bin/bash

# Quantize models using TQHD and evaluate them in fault scenarios.
# This script trains models with dfferent random seeds and then quantize them
# using TQHD choosing different number of bits. The quantized models
# are evaluated with different fault rates when making predictions. The
# purpose of this experiment is to understand how fault-tolerant is TQHD.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
POOL_DIR=_pool # Result dir to be used
MAX_SEED=20 # Max number of seeds evaluated
JOBS=11 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

start=1000
step=1000
stop=1000
DIMENSIONS=$(seq $start $step $stop)

start=2
step=1
stop=4
BITS=$(seq $start $step $stop)

start=0.01
step=0.01
stop=0.10
FAULT_RANGE=$(seq -w $start $step $stop | sed 's/,/./')

# Create the AM command based on arguments
# $1: Number of bits in expansion
function create_am_cmd() {
    local bits=$1
    local intervals=$(expr $bits + 1)
    echo "--am-type TQHD --am-bits $bits --am-intervals $intervals --am-tqhd-deviation 1.0"
}

# Create patch command if necessary.
# $1: Model pool path
# $2: Dimensions
# $3: Seed
function create_load_cmd() {
    local pool_dir=$1
    local dim=$2
    local seed=$3

    echo "--load-model $pool_dir/map/encf32-amf32/d$dim/$seed.pt --patch-model --skip-train"
}

#$1: Path to python script
#$2: Output directory of the experiments
#$3: (Optional) Path to model pool to serialize model
function launch() {
    local cmd=$1
    local acc_dir=$2
    local pool_dir=""
    if [[ $# -eq 3 ]]; then
        pool_dir=$3
    fi

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for dim in $DIMENSIONS ; do
            for bit in $BITS ; do
                for fr in $FAULT_RANGE ; do
                    vsa='--vsa MAP'
                    am_cmd=$(create_am_cmd $bit)
                    local model_name="b$bit/d$dim/$fr"
                    local acc_file="$acc_dir/$model_name/$seed.acc"
                    local fault_cmd="--am-prediction Fault --am-fault-rate $fr"

                    # Create load command if there is a pool for this experiment
                    local load_cmd=""
                    if [ $pool_dir ]; then
                        load_cmd=$(create_load_cmd $pool_dir $dim $seed)
                    fi

                    echo py_launch "$cmd $load_cmd $vsa $am_cmd $fault_cmd --vector-size $dim --device $DEVICE --seed $seed --accuracy-file $acc_file"
                done
            done
        done
    done
}
function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/tqhd"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch "src/voicehd_hdc.py" "$acc_dir" "$pool_dir"
    echo "\n"
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app/hdc/all/paper-fault/tqhd"
    launch "src/emg.py" "$acc_dir"
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/tqhd"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch "src/mnist_hdc.py" "$acc_dir" "$pool_dir"
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/tqhd"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch "src/language.py" "$acc_dir" "$pool_dir"
    echo "\n"
}

function graphhd() {
    local app="graphhd-dd"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/tqhd"
    local pool_dir="$POOL_DIR/$app/hdc"
    local dataset="DD"

    # Ensure CPU usage in this app since CUDA might consume a lot of GPU RAM
    local old_device="$DEVICE"
    DEVICE='cpu'
    launch "src/graphhd.py --dataset $dataset" "$acc_dir" "$pool_dir"
    # Restore previous device used
    DEVICE="$old_device"

    echo "\n"
}

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
