#!/bin/bash

# Quantize models using PQHDC and evaluate them in fault scenarios.
# This script trains models with dfferent random seeds and then quantize them
# using PQHDC choosing different number of projections. The quantized models
# are evaluated with different fault rates when making predictions. The
# purpose of this experiment is to understand how fault-tolerant is PQHDC.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
POOL_DIR=_pool # Result dir to be created
MAX_SEED=20 # Max number of seeds evaluated
JOBS=8 # Number of parallel jobs to be executed
DEVICE=cuda # Device used
# Choose whether model patching should always be made or not. Change it to "1"
# if you want to regenerate the patched models. Set it to empty ("") for false.
FORCE_PATCH=""

start=1000
step=1000
stop=1000
DIMENSIONS=$(seq $start $step $stop)

start=2
step=1
stop=4
PROJECTIONS=$(seq $start $step $stop)

start=0.01
step=0.01
stop=0.10
FAULT_RANGE=$(seq -w $start $step $stop | sed 's/,/./')

#$1: Path to python script
#$2: Output directory of the experiments
function launch() {
    local cmd=$1
    local acc_dir=$2

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for dim in $DIMENSIONS ; do
            for projection in $PROJECTIONS ; do
                for fr in $FAULT_RANGE ; do
                    vsa='--vsa MAP'
                    am_type="--am-type PQHDC --am-pqhdc-projections $projection"
                    local model_name="p$projection/d$dim/$fr"
                    local acc_file="$acc_dir/$model_name/$seed.acc"
                    local fault_cmd="--am-prediction Fault --am-fault-rate $fr"
                    echo py_launch "$cmd $vsa $am_type $fault_cmd --vector-size $dim --device $DEVICE --seed $seed --accuracy-file $acc_file"
                done
            done
        done
    done
}

# Create patch command if necessary.
# $1: Model path
# $2: Projections
# $3: Dimensions
# $4: Seed
# $5: Fault rate of patched model
patch_model() {
    local pool_path=$1
    local proj=$2
    local dim=$3
    local seed=$4
    local fr=$5

    model="$pool_path/pqhdc/p$proj/d$dim/$seed.pt"
    new_model="$pool_path/fault/pqhdc/p$proj/d$dim/$fr/$seed.pt"
    patch_arg="--am-prediction Fault --am-fault-rate $fr"
    # Avoid patching if there is already a patched model. This behavior can be
    # controlled by "FORCE_PATCH" variable.
    if [[ $FORCE_PATCH || ! -f $new_model ]]; then
        echo "py_launch src/patch-model.py $model $new_model $patch_arg"
    fi
}

#$1: Path to python script
#$2: Output directory of the experiments
#$3: Path to patched models dir
function launch_patched() {
    local cmd=$1
    local acc_dir=$2
    local pool_dir=$3

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for dim in $DIMENSIONS ; do
            for projection in $PROJECTIONS ; do
                for fr in $FAULT_RANGE ; do
                    am_type="--am-type PQHDC --am-pqhdc-projections $projection"
                    local model_name="p$projection/d$dim/$fr"
                    local acc_file="$acc_dir/$model_name/$seed.acc"
                    local fault_cmd="--am-prediction Fault --am-fault-rate $fr"
                    patch_cmd=$(patch_model $pool_dir $projection $dim $seed $fr)
                    prepend_cmd=""
                    if [ "$patch_cmd" ]; then
                        prepend_cmd="$patch_cmd && "
                    fi
                    patched_model="$pool_dir/fault/pqhdc/p$projection/d$dim/$fr/$seed.pt"
                    echo "$prepend_cmd py_launch $cmd --skip-train --load-model $patched_model --device $DEVICE --seed $seed --accuracy-file $acc_file"
                    #echo py_launch "$cmd $vsa $am_type $fault_cmd --vector-size $dim --device $DEVICE --seed $seed --accuracy-file $acc_file"
                done
            done
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/pqhdc"
    #launch "src/voicehd_hdc.py" "$acc_dir"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/voicehd_hdc.py" "$acc_dir" $model_dir
    echo "\n"
}

function emg() {
    local app="emg"
    local acc_dir="$RESULT_DIR/$app/hdc/all/paper-fault/pqhdc"
    launch "src/emg.py" "$acc_dir"
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/pqhdc"
    #launch "src/mnist_hdc.py " "$acc_dir"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/mnist_hdc.py" "$acc_dir" $model_dir
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/paper-fault/pqhdc"
    #launch "src/language.py " "$acc_dir"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/language.py" "$acc_dir" $model_dir
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
