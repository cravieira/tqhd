#!/bin/bash

# Quantize models using PQHDC and evaluate them in fault scenarios.
# This script trains models with dfferent random seeds and then quantize them
# using PQHDC choosing different number of projections. The quantized models
# are evaluated with different fault rates when making predictions. The
# purpose of this experiment is to understand how fault-tolerant is PQHDC.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
POOL_DIR=_pool # Result dir to be created
MAX_SEED=20 # Max number of seeds evaluated
JOBS=8 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

start=1000
step=1000
stop=1000
DIMENSIONS=$(seq $start $step $stop)

start=2
step=1
stop=4
PROJECTIONS=$(seq $start $step $stop)

start=0.01
step=0.01
stop=0.10
FAULT_RANGE=$(seq -w $start $step $stop | sed 's/,/./')

# Create patch command.
# $1: Model pool path
# $2: Projections
# $3: Dimensions
# $4: Seed
function create_load_cmd() {
    local pool_dir=$1
    local projections=$2
    local dim=$3
    local seed=$4

    echo "--load-model $pool_dir/pqhdc/p$projections/d$dim/$seed.pt --patch-model --skip-train"
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
            for projection in $PROJECTIONS ; do
                for fr in $FAULT_RANGE ; do
                    vsa='--vsa MAP'
                    am_type="--am-type PQHDC --am-pqhdc-projections $projection"
                    local model_name="p$projection/d$dim/$fr"
                    local acc_file="$acc_dir/$model_name/$seed.acc"
                    local fault_cmd="--am-prediction Fault --am-fault-rate $fr"

                    # Create load command if there is a pool for this experiment
                    local load_cmd=""
                    if [ $pool_dir ]; then
                        load_cmd=$(create_load_cmd $pool_dir $projection $dim $seed)
                    fi

                    echo py_launch "$cmd $load_cmd $vsa $am_type $fault_cmd --vector-size $dim --device $DEVICE --seed $seed --accuracy-file $acc_file"
                done
            done
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/pqhdc"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch "src/voicehd_hdc.py" "$acc_dir" $pool_dir
    echo "\n"
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app/hdc/all/paper-fault/pqhdc"
    launch "src/emg.py" "$acc_dir"
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/pqhdc"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch "src/mnist_hdc.py" "$acc_dir" "$pool_dir"
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/pqhdc"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch "src/language.py" "$acc_dir" "$pool_dir"
    echo "\n"
}

function hdchog() {
    local app="hdchog-fashionmnist"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/pqhdc"
    local pool_dir="$POOL_DIR/$app/hdc"
    local dataset="FashionMNIST"

    launch "src/hdchog.py --dataset $dataset" "$acc_dir" "$pool_dir"

    echo "\n"
}

# Launch GraphHD experiment for a given dataset.
# $1: Name of the dataset. Must be one of the dataset options avaible in
#   graphhd.py
function graphhd_dataset() {
    local dataset="$1"
    local lower_case_ds=$(com_to_lowercase "$dataset")
    local app="graphhd-$lower_case_ds"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/pqhdc"
    local pool_dir="$POOL_DIR/$app/hdc"

    launch "src/graphhd.py --dataset $dataset" "$acc_dir" "$pool_dir"

    echo "\n"
}

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
