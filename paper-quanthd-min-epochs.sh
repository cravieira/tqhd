#!/bin/bash

# Quantize MAP models to QuantHD using a high number of retraining epochs to
# evaluate when they approximate to TQHD for D=1000
# This script trains models with dfferent random seeds and then quantize them
# using QuantHD and considering different numbers of retraining rounds. The
# purpose of this experiment is to understand how the choice of the number of
# retrainings affect the accuracy of the models.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
POOL_DIR=_pool # Model pool keep serialized trained models
#JOBS=18 # Number of parallel jobs to be executed
JOBS=1 # Number of parallel jobs to be executed
DEVICE=cuda # Device used
EXP_NAME='quanthdbin-min-epochs'
# Optimization to cache the encoded vectors. This will require tons of RAM
RETRAIN_CACHE='1'

start=1000
step=1000
stop=1000
DIMENSIONS=$(seq $start $step $stop)

RETRAINING=100 # Number of retraining iterations evaluated

# Adopt learning rate equal to 0.05 as it provides best accuracy as stated in
# QuantHD's paper, Section II B.
LEARNING_RATE="--am-quanthd-alpha 0.05"

#$1: Path to python script
#$2: Output directory of the experiments
#$3: (Optional) Path to model pool to serialize model
function launch() {
    local cmd=$1
    local acc_dir=$2
    local pool_dir=""
    if [[ $# -eq 3 ]]; then
        pool_dir=$3
    fi

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for dim in $DIMENSIONS ; do
            vsa='--vsa MAP'
            # Retrain QuantHD binary for $retrain iterations and update
            # only in dataset batches instead of at each misprediction.
            local am_type="--am-type QuantHDBin --retrain-rounds $RETRAINING --am-learning Centroid $LEARNING_RATE"
            local model_name="r$RETRAINING/d$dim"
            local acc_file="$acc_dir/$model_name"
            local retrain_acc_dumper="--retrain-dump-acc $acc_dir/r"
            local retrain_acc_suf="--retrain-dump-acc-suffix /d$dim/$seed.acc "
            local retrain_accs="$retrain_acc_dumper $retrain_acc_suf"
            local save_cmd=""
            if [[ "$pool_dir" ]] ; then
                save_cmd="--save-model $pool_dir/r$RETRAINING/d$dim/$seed.pt --retrain-best"
            fi
            local cache_cmd=''
            if [[ "$RETRAIN_CACHE" ]]; then
                cache_cmd='--retrain-cache'
            fi
            echo py_launch "$cmd $vsa $cache_cmd $am_type --vector-size $dim --device $DEVICE --seed $seed $save_cmd $retrain_accs"
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/$EXP_NAME"
    launch "src/voicehd_hdc.py" "$acc_dir"
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/$EXP_NAME"
    launch "src/mnist_hdc.py " "$acc_dir"
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/$EXP_NAME"
    launch "src/language.py " "$acc_dir"
    echo "\n"
}

enable_venv
cmd=""
cmd+=$(voicehd)
cmd+=$(mnist)
cmd+=$(language)

#printf "$cmd"
parallel_launch "$JOBS" "$cmd"

disable_venv

