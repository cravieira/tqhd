#!/bin/bash

# Quantize models using TQHD and evaluate them in fault scenarios.
# This script trains models with dfferent random seeds and then quantize them
# using TQHD choosing different number of bits. The quantized models
# are evaluated with different fault rates when making predictions. The
# purpose of this experiment is to understand how fault-tolerant is TQHD.

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

# Launch TQHD compression experiments. Thi function loads a trained MAP model
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
                echo "py_launch src/compaction.py $map_model -c $compaction --csv $csv_path --patch-model --am-type TQHD --am-bits $bit --am-intervals $ints --am-tqhd-deviation 1.0"
            done
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction/tqhd"
    #launch "src/voicehd_hdc.py" "$acc_dir"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/voicehd_hdc.py" "$acc_dir" "$model_dir" exp_table[@]
    echo "\n"
}

# Create patch command if necessary.
# $1: Model path
# $2: Number of bits used in expansion
# $3: Dimensions
# $4: Seed
function patch_model_emg() {
    local pool_path=$1
    local bit=$2
    local dim=$3
    local seed=$4
    local cmd=$5
    local subject=$6

    model="$pool_path/map/encf32-amf32/d$dim/$seed.pt"
    new_model="$pool_path/tqhd/b$bit/d$dim/$seed.pt"
    local am_args=$(create_am_cmd $bit)
    patch_arg="$am_args"
    # Avoid patching if there is already a patched model. This behavior can be
    # controlled by "FORCE_PATCH" variable.
    if [[ $FORCE_PATCH || ! -f $new_model ]]; then
        ints=$(expr $bit + 1)
        echo "py_launch $cmd --subject $subject --vector-size $dim --am-type TQHD --am-tqhd-deviation 1.0 --am-bits $bit --am-intervals $ints --seed $seed --save-model $new_model --device $DEVICE"
    fi
}

# Patch all models necessary for EMG compression estimation.
#$1: Path to python script
#$2: Output directory of the experiments
#$3: Path to patched models dir
#$4: Pointer to experiment table
function patch_emg() {
    local cmd=$1
    local acc_dir=$2
    local pool_dir=$3
    local -a table=("${!4}")

    for (( subject = 0; subject < 5; subject++)); do
        for (( seed = 0; seed < $MAX_SEED; seed++ )); do
            for dim in $DIMENSIONS ; do
                for (( i = 0; i < ${#table[@]}; i++ )); do
                    # Split the experiment table into its two variables
                    IFS=' ' read -r bit compaction <<< ${table[i]}
                    patch_cmd=$(patch_model_emg "$pool_dir-s$subject" $bit $dim $seed "$cmd" $subject)
                    if [ "$patch_cmd" ]; then
                        echo "$patch_cmd"
                    fi
                done
            done
        done
    done
}

# Launch compression experiments for emg.
# Each emg model is trained for its correspondent dataset subject, quantized,
# and serialized to file. Then, the serialized model is used for compression.
#$1: Path to python script
#$2: Output directory of the experiments
#$3: Path to patched models dir
#$4: Pointer to experiment table
function launch_patched_emg() {
    local cmd=$1
    local acc_dir=$2
    local pool_dir=$3
    local -a table=("${!4}")

    # Train and serialize all require emg models first before using them.
    patch_emg "$cmd" "$acc_dir" "$pool_dir" table[@]

    for (( subject = 0; subject < 5; subject++)); do
        for (( seed = 0; seed < $MAX_SEED; seed++ )); do
            for dim in $DIMENSIONS ; do
                for (( i = 0; i < ${#table[@]}; i++ )); do
                    # Split the experiment table into its two variables
                    IFS=' ' read -r bit compaction <<< ${table[i]}

                    local patched_model="$pool_dir/emg-s$subject/tqhd/b$bit/d$dim"
                    local csv_path="$acc_dir-s$subject/$patched_model/c$compaction/$seed.csv"
                    echo "py_launch src/compaction.py $patched_model -c $compaction --csv $csv_path"
                done
            done
        done
    done
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app"
    local model_dir=$POOL_DIR/$app
    launch_patched_emg "src/emg.py" "$acc_dir" "$model_dir" exp_table[@]
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction/tqhd"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/mnist_hdc.py" "$acc_dir" "$model_dir" exp_table[@]
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction/tqhd"
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
printf "$cmd" | parallel --verbose -j$JOBS --halt now,fail=1
disable_venv

