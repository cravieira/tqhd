#!/bin/bash

# Accuracy of MAP models with random seeds
# Train the available models using different random seeds. The purpose of this
# experiment is to evaluate how the accuracy of the models change with
# different initial values and dimensions

set -e

source common.sh

RESULTS_DIR=_accuracy # Result dir to be created
POOL_DIR=_pool # Model pool keep serialized trained models
MAX_SEED=20 # Max number of seeds evaluated
JOBS=8 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

exp_table=(
    'map f32 f32 1000'
    'map f32 f32 2000'
    'map f32 f32 3000'
    'map f32 f32 4000'
    'map f32 f32 5000'
    'map f32 f32 6000'
    'map f32 f32 7000'
    'map f32 f32 8000'
    'map f32 f32 9000'
    'map f32 f32 10000'
)

# Create the command line used in HDC models.
# $1: vsa type.
# $2: data type used by encoding.
# $3: data type used by associative memory.
# $4: vector size.
function parse_parameters() {
    local vsa=$1
    local enc=$2
    local am=$3
    local dim=$4
    # Using ${vsa^^} to invert it to upper case
    echo "--vsa ${vsa^^} --dtype-enc $enc --dtype-am $am --vector-size $dim"
}

#$1: Path to python script
#$2: Accuracy directory of the experiments
#$3: Pointer to the parameter table
#$4: (Optional) Path to model pool to serialize model
function launch_hdc_table() {
    local cmd=$1
    local acc_dir=$2
    declare -a table=("${!3}")
    local pool_dir=$4

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for (( i=0; i < ${#table[@]}; i++ )) ; do
            # Split the experiment table into four variables
            IFS=' ' read -r vsa enc am dim <<< ${table[i]}
            local exp=$(parse_parameters ${table[i]})
            local model_name=$vsa-enc$enc-am$am-d$dim
            local acc_file="$acc_dir/$model_name/$seed.acc"
            local save_cmd=""
            if [[ "$pool_dir" ]] ; then
                local model_file="$"
                save_cmd="--save-model $pool_dir/$vsa/enc$enc-am$am/d$dim/$seed.pt"
            fi
            echo py_launch "$cmd $exp --device $DEVICE --seed $seed --accuracy-file $acc_file $save_cmd"
        done
    done
}

function voicehd() {
    local app="voicehd"

    # HDC
    local acc_dir="$RESULTS_DIR/$app/hdc"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_hdc_table "src/voicehd_hdc.py" "$acc_dir" exp_table[@] "$model_dir"

    echo "\n"
}

function emg() {
    local app="emg"

    # Run EMG on all subjects available in the dataset
    local acc_dir="$RESULTS_DIR/$app/hdc/all"
    launch_hdc_table "src/emg.py" "$acc_dir" exp_table[@]

    echo "\n"
}

function mnist() {
    local app="mnist"

    local acc_dir="$RESULTS_DIR/$app/hdc"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_hdc_table "src/mnist_hdc.py " "$acc_dir" exp_table[@] "$model_dir"

    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULTS_DIR/$app/hdc"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_hdc_table "src/language.py " "$acc_dir" exp_table[@] "$model_dir"

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
