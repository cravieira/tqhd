#!/bin/bash

# Quantize models using TQHD with a fixed deviation value but vary intervals
# and number of bits.
# This script trains models with dfferent random seeds and then quantize them
# using AM Thermometer choosing different number of intervals and bits as
# quantization parameters. The purpose of this experiment is to understand how
# the different intervals and bits parameter choices affect quantization.

set -e

source common.sh

RESULT_DIR=_transformation # Result dir to be created
MAX_SEED=20 # Max number of seeds evaluated
JOBS=6 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

# Experiment tables
# <Intervals> <Bits> <Dimension>
exp_table=(
    # AssociativeMemory T
    '2  2 1000'
    '3  2 1000'
    #3  3 1000' #Exclude odd intervals if they are not equal to bits+1
    '4  3 1000'
    #'3  4 1000'
    '4  4 1000'
    '5  4 1000'
    #'3  5 1000'
    '4  5 1000'
    #'5  5 1000'
    '6  5 1000'
    #'3  6 1000'
    '4  6 1000'
    #'5  6 1000'
    '6  6 1000'
    '7  6 1000'
    #'3  7 1000'
    '4  7 1000'
    #'5  7 1000'
    '6  7 1000'
    #'7  7 1000'
    '8  7 1000'
    )

# Create the command line used in HDC script models.
# $1: Number of intervals.
# $2: Number of bits to expand to.
# $3: Vector size.
function parse_parameters() {
    local ints=$1
    local bits=$2
    local dim=$3
    echo "--vector-size $dim --am-intervals $ints --am-bits $bits"
}

#$1: Path to python script
#$2: Output directory of the experiments
#$3: Pointer to the parameter table
function launch_hdc_table() {
    local cmd=$1
    local acc_dir=$2
    declare -a table=("${!3}")

    # Fixed standard deviation
    dev=1.0

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for (( i=0; i < ${#table[@]}; i++ )) ; do
            vsa='--vsa MAP'
            am_type='--am-type TD'
            # Split the experiment table into its two variables
            IFS=' ' read -r intervals bits dim <<< ${table[i]}
            local exp=$(parse_parameters ${table[i]})
            local model_name="amtd-std$dev-ints$intervals-bits$bits-d$dim"
            local acc_file="$acc_dir/$model_name/$seed.acc"
            echo py_launch "$cmd $exp $vsa $am_type --am-td-deviation $dev --device $DEVICE --seed $seed --accuracy-file $acc_file"
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-int-bits"
    launch_hdc_table "src/voicehd_hdc.py" "$acc_dir" exp_table[@]
    echo "\n"
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app/hdc/all/paper-int-bits"
    launch_hdc_table "src/emg.py" "$acc_dir" exp_table[@]
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-int-bits"
    launch_hdc_table "src/mnist_hdc.py " "$acc_dir" exp_table[@]
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-int-bits"
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
printf "$cmd" | parallel --verbose -j$JOBS #--halt now,fail=1
disable_venv
