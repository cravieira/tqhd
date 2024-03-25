#!/bin/bash

# Quantize MAP models to PQHDC using different numbers of projections.
# This script trains models with dfferent random seeds and then quantize them
# using PQHDC and different numbers of projections. The purpose of this
# experiment is to understand how the choice of the number of projections
# affect the accuracy of the models.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
POOL_DIR=_pool # Model pool keep serialized trained models
MAX_SEED=20 # Max number of seeds evaluated
JOBS=10 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

start=1000
step=1000
stop=10000
DIMENSIONS=$(seq $start $step $stop)

start=2
step=1
stop=4
PROJECTIONS=$(seq $start $step $stop)

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
            for projection in $PROJECTIONS ; do
                vsa='--vsa MAP'
                am_type="--am-type PQHDC --am-pqhdc-projections $projection"
                local model_name="p$projection/d$dim"
                local acc_file="$acc_dir/$model_name/$seed.acc"
                save_cmd=""
                if [[ "$pool_dir" ]] ; then
                    local model_file="$"
                    save_cmd="--save-model $pool_dir/p$projection/d$dim/$seed.pt"
                fi
                echo py_launch "$cmd $vsa $am_type --vector-size $dim --device $DEVICE --seed $seed --accuracy-file $acc_file $save_cmd"
            done
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/pqhdc"
    local model_dir="$POOL_DIR/$app/hdc/pqhdc"
    launch "src/voicehd_hdc.py" "$acc_dir" "$model_dir"
    echo "\n"
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app/hdc/all/pqhdc"
    launch "src/emg.py" "$acc_dir"
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/pqhdc"
    local model_dir="$POOL_DIR/$app/hdc/pqhdc"
    launch "src/mnist_hdc.py " "$acc_dir" "$model_dir"
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/pqhdc"
    local model_dir="$POOL_DIR/$app/hdc/pqhdc"
    launch "src/language.py " "$acc_dir" "$model_dir"
    echo "\n"
}

function hdchog() {
    local app="hdchog-fashionmnist"
    local acc_dir="$RESULT_DIR/$app/hdc/pqhdc"
    local pool_dir="$POOL_DIR/$app/hdc/pqhdc"
    local dataset="FashionMNIST"

    launch "src/hdchog.py --dataset $dataset" "$acc_dir" "$pool_dir"

    echo "\n"
}

function graphhd() {
    local app="graphhd-dd"
    local acc_dir="$RESULT_DIR/$app/hdc/pqhdc"
    local pool_dir="$POOL_DIR/$app/hdc/pqhdc"
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
cmd+=$(hdchog)
cmd+=$(graphhd)

#printf "$cmd"
parallel_launch "$JOBS" "$cmd"
disable_venv
