#!/bin/bash

# Quantize MAP models to QuantHD using different numbers retraining iterations.
# This script trains models with dfferent random seeds and then quantize them
# using QuantHD and considering different numbers of retraining rounds. The
# purpose of this experiment is to understand how the choice of the number of
# retrainings affect the accuracy of the models.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
POOL_DIR=_pool # Model pool keep serialized trained models
MAX_SEED=20 # Max number of seeds evaluated
JOBS=18 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

start=1000
step=1000
stop=10000
DIMENSIONS=$(seq $start $step $stop)

RETRAINING=15 # Number of retraining iterations evaluated

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
            #TODO Ajeitar o suffix e o dumper. Passar o dim para o suffix
            local retrain_accs="$retrain_acc_dumper $retrain_acc_suf"
            local save_cmd=""
            if [[ "$pool_dir" ]] ; then
                save_cmd="--save-model $pool_dir/r$RETRAINING/d$dim/$seed.pt --retrain-best"
            fi
            echo py_launch "$cmd $vsa $am_type --vector-size $dim --device $DEVICE --seed $seed $save_cmd $retrain_accs"
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/quanthdbin"
    local model_dir="$POOL_DIR/$app/hdc/quanthdbin"
    launch "src/voicehd_hdc.py" "$acc_dir" "$model_dir"
    echo "\n"
}

function emg() {
    local app="emg"
    # Run experiments individually for each subject available
    local subjects=$(seq 0 4)
    for s in $subjects; do
        local acc_dir="$RESULT_DIR/$app-s$s/hdc/quanthdbin"
        launch "src/emg.py --subject $s" "$acc_dir"
        echo "\n"
    done
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/quanthdbin"
    local model_dir="$POOL_DIR/$app/hdc/quanthdbin"
    launch "src/mnist_hdc.py " "$acc_dir" "$model_dir"
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/quanthdbin"
    local model_dir="$POOL_DIR/$app/hdc/quanthdbin"
    # Disable in-memory cache of training dataset since the dataset used in
    # this experiment is too big.
    launch "src/language.py " "$acc_dir" "$model_dir"
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
