#!/bin/bash

# Accuracy of MAP models with random seeds
# Train the available models using different random seeds. The purpose of this
# experiment is to evaluate how the accuracy of the models change with
# different initial values and dimensions

set -e

source common.sh

RESULTS_DIR=_accuracy # Result dir to be created
POOL_DIR=_pool # Model pool keep serialized trained models
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
    # The implementation of tis function is slightly different from the other
    # apps as EMG's dataset are split into 5 subjects. When running emg.py
    # without the --subject argument, the script trains all five subjects and
    # returns the mean accuracy. Since EMG is a lightweight model that can be
    # trained cheapily, this function first trains the "-all" version to obtain
    # the mean accuracies and then train each subject individually to serialize
    # EMG models.
    local app="emg"

    # Run EMG on all subjects available in the dataset. The results of
    # "emg-all" contain the mean of all subjects.
    local acc_dir="$RESULTS_DIR/$app-all/hdc"
    #launch_hdc_table "src/emg.py" "$acc_dir" exp_table[@]

    # Train a model for each subject, serialize it to disk, and save its
    # accuracy in a app-s<number> directory.
    local subjects=$(seq 0 4)
    for s in $subjects; do
        local acc_dir="$RESULTS_DIR/$app-s$s/hdc"
        local model_dir="$POOL_DIR/$app-s$s/hdc"
        launch_hdc_table "src/emg.py --subject $s" "$acc_dir" exp_table[@] "$model_dir"
        echo "\n"
    done

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

function hdchog() {
    local app="hdchog-fashionmnist"
    local acc_dir="$RESULTS_DIR/$app/hdc"
    local model_dir="$POOL_DIR/$app/hdc"
    local dataset="FashionMNIST"
    launch_hdc_table "src/hdchog.py --dataset $dataset" "$acc_dir" exp_table[@] "$model_dir"

    echo "\n"
}

# Launch GraphHD experiment for a given dataset.
# $1: Name of the dataset. Must be one of the dataset options avaible in
#   graphhd.py
function graphhd_dataset() {
    local dataset="$1"
    local lower_case_ds=$(com_to_lowercase "$dataset")
    local app="graphhd-$lower_case_ds"
    local acc_dir="$RESULTS_DIR/$app/hdc"
    local model_dir="$POOL_DIR/$app/hdc"

    launch_hdc_table "src/graphhd.py --dataset $dataset" "$acc_dir" "exp_table[@]" "$model_dir"
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
