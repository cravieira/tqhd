#!/bin/bash

# Evaluate TQHD compression capabilities
# This script trains models with dfferent random seeds and then quantize them
# using TQHD choosing different number of bits. The quantized models
# are evaluated with different compression parameters. The results are saved in
# a .csv file. The purpose of this experiment is to assess the TQHD compression
# when using different parameters and matrices.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
POOL_DIR=_pool # Result dir to be created
MAX_SEED=20 # Max number of seeds evaluated
JOBS=10 # Number of parallel jobs to be executed
DEVICE=cuda # Device used
# Choose whether model patching should always be made or not. Change it to "1"
# if you want to regenerate the patched models. Set it to empty ("") for false.
FORCE_PATCH=""

start=1000
step=1000
stop=10000
DIMENSIONS=$(seq $start $step $stop)
# Experiment tables
# <Bits> <Compaction>
exp_table=(
    '3 3'
    '4 3'
    '4 4'
    '5 4'
    '6 4'
    '7 4'
    '8 4'
    )

# Create the AM command based on arguments
# $1: Number of bits in expansion
function create_am_cmd() {
    local bits=$1
    local intervals=$(expr $bits + 1)
    echo "--am-type TQHD --am-bits $bits --am-intervals $intervals --am-tqhd-deviation 1.0"
}

# Launch TQHD compression experiments. This function loads a trained MAP model
# and converts it to a TQHD model before running compaction.
#$1: Path to python script
#$2: Output directory of the experiments
#$3: Path to patched models dir
#$4: Pointer to experiment table
function launch_patched() {
    local cmd=$1
    local acc_dir=$2
    local pool_dir=$3
    local -a table=("${!4}")

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for dim in $DIMENSIONS ; do
            for (( i = 0; i < ${#table[@]}; i++ )); do
                # Split the experiment table into its two variables
                IFS=' ' read -r bit compaction <<< ${table[i]}
                local ints=$(expr $bit + 1)

                local model_name="b$bit/d$dim"
                local acc_file="$acc_dir/$model_name/$seed.acc"
                local csv_path="$acc_dir/$model_name/c$compaction/$seed.csv"

                # Load trained MAP model and patch it, i.e., quantize it, to
                # TQHD before running compaction.
                # Quantize a trained MAP model to TQHD and estimate its compaction.
                local map_model="$pool_dir/map/encf32-amf32/d$dim/$seed.pt"
                echo "py_launch src/compaction.py $map_model -c0 $compaction -c1 $compaction --am-tqhd-encode-table BaseZero --csv $csv_path --patch-model --am-type TQHD --am-bits $bit --am-intervals $ints --am-tqhd-deviation 1.0"
            done
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction/tqhd/BaseZero"
    #launch "src/voicehd_hdc.py" "$acc_dir"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/voicehd_hdc.py" "$acc_dir" "$model_dir" exp_table[@]
    echo "\n"
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app"
    local pool_dir="$POOL_DIR/$app"

    local subjects=$(seq 0 4)
    for s in $subjects; do
        local acc_dir="$RESULT_DIR/$app-s$s/hdc/compaction/BaseZero"
        local pool_dir="$POOL_DIR/$app-s$s/hdc"
        launch_patched "src/emg.py --subject $s" "$acc_dir" "$pool_dir" exp_table[@]
        echo "\n"
    done

    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction/BaseZero"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/mnist_hdc.py" "$acc_dir" "$model_dir" exp_table[@]
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction/BaseZero"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/language.py" "$acc_dir" "$model_dir" exp_table[@]
    echo "\n"
}

enable_venv
cmd=""
cmd+=$(voicehd)
cmd+=$(emg)
cmd+=$(mnist)
cmd+=$(language)

#printf "$cmd"
parallel_launch "$JOBS" "$cmd"
disable_venv

