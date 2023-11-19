#!/bin/bash

# Accuracy with random seed experiment
# Train the available models using different random seeds. The purpose of this
# experiment is to evaluate how the accuracy of the models change with
# different initial values.

set -e

source common.sh

RESULTS_DIR=_accuracy # Result dir to be created
POOL_DIR=_pool # Model pool keep serialized trained models
MAX_SEED=20 # Max number of seeds evaluated
JOBS=7 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

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

#$1: Path to python script
#$2: Output directory of the experiments
function launch_nn() {
    local cmd=$1
    local acc_dir=$2
    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        acc_file="$acc_dir/$seed.acc"
        echo py_launch "$cmd --seed $seed --accuracy-file $acc_file"
    done
}

function voicehd() {
    local app="voicehd"

    # HDC
    local acc_dir="$RESULTS_DIR/$app/hdc"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_hdc_table "src/voicehd_hdc.py" "$acc_dir" voicehd_hdc_exp[@] "$model_dir"

    # Neural Network
    acc_dir="$RESULTS_DIR/$app/nn"
    launch_nn "src/voicehd_nn.py" $acc_dir

    echo "\n"
}

function emg() {
    local app="emg"

    # Run EMG on all subjects available in the dataset
    local acc_dir="$RESULTS_DIR/$app/hdc/all"
    launch_hdc_table "src/emg.py" "$acc_dir" emg_hdc_exp[@]

    echo "\n"
}

function mnist() {
    local app="mnist"

    local acc_dir="$RESULTS_DIR/$app/hdc"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_hdc_table "src/mnist_hdc.py " "$acc_dir" mnist_hdc_exp[@] "$model_dir"

    # Neural Network
    acc_dir="$RESULTS_DIR/$app/lenet"
    launch_nn "src/mnist_lenet.py" $acc_dir

    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULTS_DIR/$app/hdc"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_hdc_table "src/language.py " "$acc_dir" language_hdc_exp[@] "$model_dir"

    echo "\n"
}

enable_venv
cmd=""
cmd+=$(voicehd)
cmd+=$(emg)
cmd+=$(mnist)
cmd+=$(language)

printf "$cmd"
#printf "$cmd" | parallel --verbose -j$JOBS --halt now,fail=1
disable_venv

# Export generated directory to the project's root folder so that other parts
# of the repository can access it.
ln -fs ./train/$RESULTS_DIR ../
ln -fs ./train/$POOL_DIR ../
