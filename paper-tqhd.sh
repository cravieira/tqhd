#!/bin/bash

# Quantize models using TQHD with fixed deviation, different number of bits and
# always max number of intervals.
# This script trains models with dfferent random seeds and then quantize them
# using AM Thermometer choosing different number of intervals and bits as
# quantization parameters. the purpose of this experiment is to understand how
# the different intervals and bits parameter choices affect quantization.

# How does this script work?
# This script works by patching a trained MAP float model to TQHD to avoid the
# burden of retraining the models again. This strategy significantly reduces
# the script execution time. However, for EMG the training is executed again
# since it is a "multidataset" and has one model for each subject dataset.
# Fortunately, its training time is very low compared to the other apps.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
POOL_DIR=_pool # Pool dir to be created
MAX_SEED=20 # Max number of seeds evaluated
JOBS=16 # Number of parallel jobs to be executed
DEVICE=cuda # Device used
start=1000
step=1000
final=10000
dimensions=$(seq $start $step $final)

# <Bits>
exp_table=(
    '2'
    '3'
    '4'
    '5'
    '6'
    '7'
    '8'
)

# Create the command line used in HDC script models based on the experiment
# table.
# $1: Bits
function parse_parameters() {
    local bits=$1
    local ints=$(expr $bits + 1)
    echo "--am-intervals $ints --am-bits $bits"
}

# Create patch command if necessary.
# $1: Model pool path
# $2: Dimensions
# $3: Seed
function create_load_cmd() {
    local pool_dir=$1
    local dim=$2
    local seed=$3

    echo "--load-model $pool_dir/map/encf32-amf32/d$dim/$seed.pt --patch-model --skip-train"
}

#$1: Path to python script
#$2: Output directory of the experiments
#$3: Path to patched models dir
#$4: [Optional] Path to pool dir. If given, then patch model from the trained
#MAP model pool and evaluates the quantized model on the test dataset.
function launch() {
    local cmd=$1
    local acc_dir=$2
    declare -a table=("${!3}")
    local pool_dir=''
    if [[ $# -eq 4 ]]; then
        pool_dir=$4
    fi

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for dim in $dimensions ; do
            for (( i=0; i < ${#table[@]}; i++ )) ; do
                vsa='--vsa MAP'
                am_type='--am-type TQHD'
                # Get the variable in experiment table
                IFS=' ' read -r bits <<< ${table[i]}
                local exp=$(parse_parameters ${table[i]})
                local model_name="d$dim"
                local acc_file="$acc_dir/b$bits/$model_name/$seed.acc"

                # Create load command if there is a pool for this experiment
                local load_cmd=""
                if [ $pool_dir ]; then
                    load_cmd=$(create_load_cmd $pool_dir $dim $seed)
                fi

                echo py_launch "$cmd $exp $load_cmd $vsa $am_type --vector-size $dim --device $DEVICE --seed $seed --accuracy-file $acc_file"
            done
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-tqhd"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch "src/voicehd_hdc.py" "$acc_dir" exp_table[@] "$pool_dir"
    echo "\n"
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app/hdc/all/paper-tqhd"
    launch "src/emg.py" "$acc_dir" exp_table[@]
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-tqhd"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch "src/mnist_hdc.py " "$acc_dir" exp_table[@] "$pool_dir"
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-tqhd"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch "src/language.py " "$acc_dir" exp_table[@] "$pool_dir"
    echo "\n"
}

function graphhd() {
    local app="graphhd-dd"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-tqhd"
    local pool_dir="$POOL_DIR/$app/hdc"
    local dataset="DD"

    # Ensure CPU usage in this app since CUDA might consume a lot of GPU RAM
    local old_device="$DEVICE"
    DEVICE='cpu'
    launch "src/graphhd.py --dataset $dataset" "$acc_dir" exp_table[@] "$pool_dir"
    # Restore previous device used
    DEVICE="$old_device"

    echo "\n"
}

enable_venv
cmd=""
cmd+=$(voicehd)
cmd+=$(emg)
cmd+=$(mnist)
cmd+=$(language)
cmd+=$(graphhd)

#printf "$cmd"
parallel_launch "$JOBS" "$cmd"
disable_venv
