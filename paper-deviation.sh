#!/bin/bash

# Quantize models using TQHD with different deviation values.
# This script trains models with dfferent random seeds and then quantize them
# using AM Thermometer choosing the quantization poles according to the
# deviation of the model. The models are experimented choosing a variable
# number of dimension expansion, i.e., 1 dimension MAP is expanded to different
# number of dimensions in the quantized vector. The number of quantization
# intervals is always the maximal number possible, i.e., <bits>+1. The purpose
# of this experiment is to understand how the accuracy of quantized models vary
# with different quantization values based on the standard deviation.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
MAX_SEED=20 # Max number of seeds evaluated
JOBS=11 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

# Experiment tables
# <Bits> <Dimension>
exp_table=(
    '2 1000'
    '3 1000'
    '4 1000'
    '5 1000'
    '6 1000'
    '7 1000'
    '8 1000'
)

start=0.1
step=0.1
final=2.0
std_deviation=$(seq -w $start $step $final | sed 's/,/./')

# Create the command line used in HDC script models.
# $1: Number of bits to expand to.
# $2: Vector size.
function parse_parameters() {
    local bits=$1
    local dim=$2
    local ints=$(expr $bits + 1)
    echo "--vector-size $dim --am-intervals $ints --am-bits $bits"
}

#$1: Path to python script
#$2: Output directory of the experiments
#$3: Pointer to the parameter table
function launch_hdc_table() {
    local cmd=$1
    local acc_dir=$2
    declare -a table=("${!3}")
    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for dev in $std_deviation ; do
            for (( i=0; i < ${#table[@]}; i++ )) ; do
                vsa='--vsa MAP'
                am_type='--am-type TD'
                # Split the experiment table into its two variables
                IFS=' ' read -r bits dim <<< ${table[i]}
                local exp=$(parse_parameters ${table[i]})
                local model_name="amtd-std$dev-bits$bits-d$dim"
                local acc_file="$acc_dir/$model_name/$seed.acc"
                echo py_launch "$cmd $exp $vsa $am_type --am-tqhd-deviation $dev --device $DEVICE --seed $seed --accuracy-file $acc_file"
            done
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/deviation"
    launch_hdc_table "src/voicehd_hdc.py" "$acc_dir" exp_table[@]
    echo "\n"
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app/hdc/all/deviation"
    launch_hdc_table "src/emg.py" "$acc_dir" exp_table[@]
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/deviation"
    launch_hdc_table "src/mnist_hdc.py " "$acc_dir" exp_table[@]
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/deviation"
    launch_hdc_table "src/language.py " "$acc_dir" exp_table[@]
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

# Export generated directory to the project's root folder so that other parts
# of the repository can access it.
#ln -fs ./train/$RESULT_DIR ../

