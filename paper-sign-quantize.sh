#!/bin/bash

# Quantize models using Sign Quantization
# This script trains models with dfferent random seeds and then quantize them
# using Sign Quantization. The purpose of this experiment is to understand how
# the this naive quantization approach affects accuracy of the models.

set -e

source common.sh

RESULT_DIR=_transformation # Result dir to be created
MAX_SEED=20 # Max number of seeds evaluated
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
    local acc_dir="$RESULT_DIR/$app/hdc/all/signquantize"
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

enable_venv
cmd=""
cmd+=$(voicehd)
cmd+=$(emg)
cmd+=$(mnist)
cmd+=$(language)

#printf "$cmd"
printf "$cmd" | parallel --verbose -j$JOBS --halt now,fail=1
disable_venv

