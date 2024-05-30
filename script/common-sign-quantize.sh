#!/bin/bash

# Quantize models using Sign Quantization
# This script trains models with dfferent random seeds and then quantize them
# using Sign Quantization. The purpose of this experiment is to understand how
# the this naive quantization approach affects accuracy of the models.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
JOBS=10 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

start=1000
step=1000
stop=1000
DIMENSIONS=$(seq $start $step $stop)

#$1: Path to python script
#$2: Output directory of the experiments
function launch() {
    local cmd=$1
    local acc_dir=$2

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for dim in $DIMENSIONS ; do
            vsa='--vsa MAP'
            am_type='--am-type SQ'
            # Split the experiment table into its two variables
            local model_name="amsq-d$dim"
            local acc_file="$acc_dir/$model_name/$seed.acc"
            echo py_launch "$cmd $vsa $am_type --vector-size $dim --device $DEVICE --seed $seed --accuracy-file $acc_file"
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/signquantize"
    launch "src/voicehd_hdc.py" "$acc_dir"
    echo "\n"
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app-all/hdc/signquantize"
    launch "src/emg.py" "$acc_dir"
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/signquantize"
    launch "src/mnist_hdc.py " "$acc_dir"
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/signquantize"
    launch "src/language.py " "$acc_dir"
    echo "\n"
}

function hdchog() {
    local app="hdchog-fashionmnist"
    local acc_dir="$RESULT_DIR/$app/hdc/signquantize"
    local dataset="FashionMNIST"

    launch "src/hdchog.py --dataset $dataset" "$acc_dir"

    echo "\n"
}

# Launch GraphHD experiment for a given dataset.
# $1: Name of the dataset. Must be one of the dataset options avaible in
#   graphhd.py
function graphhd_dataset() {
    local dataset="$1"
    local lower_case_ds=$(com_to_lowercase "$dataset")
    local app="graphhd-$lower_case_ds"
    local acc_dir="$RESULT_DIR/$app/hdc/signquantize"

    launch "src/graphhd.py --dataset $dataset" "$acc_dir"

    echo "\n"
}
