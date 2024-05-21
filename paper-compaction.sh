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
JOBS=17 # Number of parallel jobs to be executed
DEVICE=cuda # Device used

start=1000
step=1000
stop=10000
DIMENSIONS=$(seq $start $step $stop)
# Experiment tables
# The table is organized in 5 columns as defined below:
# <TQHD Encoding Table> <TQHD Bits> <Symbol 0 size> < Symbol 1 size> <interleave: yes|no>
# The entries are split in 3 sections:
# - Section 1: The first section runs experiments with BaseZero no interleave to
# show that interleaving is necessary for this encoding. However, the
# compression results are worse than interleaving and they should be
# descondired. However, running these experiments allow us to reason about the
# increase of the mean group length, as we claim interleaving is necessary.
# Since the script compaction.py outputs the mean group length for each symbol
# in its result csv file and this is not affected by the symbols sizes for 0
# and 1 (symbol sizes only affect compression), the table contains only one
# entry per TQHD expansion bit evaluated and the symbols sizes are kept for
# compatibility with launch patched.
# - Section 2: Broad evaluation of triangular matrix with compression
# interleaving to assess its capabilities.
# - Section 3: Evaluation of BandMatrix without interleaving as the enconding
# table is already interleaved.
exp_table=(
    # Section 1: BaseZero (Triangular Matrix) no interleaving
    'BaseZero   3 2 2 no'
    'BaseZero   4 2 2 no'
    'BaseZero   5 2 2 no'
    'BaseZero   6 2 2 no'
    'BaseZero   7 2 2 no'
    'BaseZero   8 2 2 no'

    # Section 2: BaseZero with interleaving
    'BaseZero   3 3 3 yes'
    'BaseZero   4 3 3 yes'
    'BaseZero   4 4 4 yes'
    'BaseZero   5 4 4 yes'
    'BaseZero   6 4 4 yes'
    'BaseZero   7 4 4 yes'
    'BaseZero   8 4 4 yes'

    # Section 3: BandMatrix without interleaving
    'BandMatrix 2 2 1 no'
    'BandMatrix 3 3 1 no'
    'BandMatrix 3 3 2 no'
    'BandMatrix 4 3 2 no'
    'BandMatrix 4 3 3 no'
    'BandMatrix 4 4 4 no'
    'BandMatrix 5 3 3 no'
    'BandMatrix 5 4 2 no'
    'BandMatrix 5 4 3 no'
    'BandMatrix 5 4 4 no'
    'BandMatrix 6 4 2 no'
    'BandMatrix 6 4 3 no'
    'BandMatrix 6 4 4 no'
    'BandMatrix 7 4 2 no'
    'BandMatrix 7 4 3 no'
    'BandMatrix 7 4 4 no'
    'BandMatrix 8 4 2 no'
    'BandMatrix 8 4 3 no'
    'BandMatrix 8 4 4 no'
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
                IFS=' ' read -r tqhd_encode tqhd_bits c0size c1size interleave <<< ${table[i]}
                local ints=$(expr $tqhd_bits + 1)

                local interleave_cmd=""
                local interleave_path="interleave"
                if [[ "$interleave" == "no" ]]; then
                    interleave_cmd="--no-interleave"
                    interleave_path="no-interleave"
                fi

                local model_name="$tqhd_encode-$interleave_path/b$tqhd_bits/d$dim"
                local acc_file="$acc_dir/$model_name/$seed.acc"
                local csv_path="$acc_dir/$model_name/c${c0size}c${c1size}/$seed.csv"

                # Load trained MAP model and patch it, i.e., quantize it, to
                # TQHD before running compaction.
                # Quantize a trained MAP model to TQHD and estimate its compaction.
                local map_model="$pool_dir/map/encf32-amf32/d$dim/$seed.pt"
                echo "py_launch src/compaction.py $map_model -c0 $c0size -c1 $c1size $interleave_cmd --am-tqhd-encode-table $tqhd_encode --csv $csv_path --patch-model --am-type TQHD --am-bits $tqhd_bits --am-intervals $ints --am-tqhd-deviation 1.0"
            done
        done
    done
}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction"
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
        local acc_dir="$RESULT_DIR/$app-s$s/hdc/compaction"
        local pool_dir="$POOL_DIR/$app-s$s/hdc"
        launch_patched "src/emg.py --subject $s" "$acc_dir" "$pool_dir" exp_table[@]
        echo "\n"
    done

    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/mnist_hdc.py" "$acc_dir" "$model_dir" exp_table[@]
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction"
    local model_dir="$POOL_DIR/$app/hdc"
    launch_patched "src/language.py" "$acc_dir" "$model_dir" exp_table[@]
    echo "\n"
}

function hdchog() {
    local app="hdchog-fashionmnist"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction"
    local pool_dir="$POOL_DIR/$app/hdc"
    local dataset="FashionMNIST"
    launch_patched "src/hdchog.py --dataset $dataset" "$acc_dir" "$pool_dir" exp_table[@]
    echo "\n"
}

# Launch GraphHD experiment for a given dataset.
# $1: Name of the dataset. Must be one of the dataset options avaible in
#   graphhd.py
function graphhd_dataset() {
    local dataset="$1"
    local lower_case_ds=$(com_to_lowercase "$dataset")
    local app="graphhd-$lower_case_ds"
    local acc_dir="$RESULT_DIR/$app/hdc/compaction"
    local pool_dir="$POOL_DIR/$app/hdc"

    launch_patched "src/graphhd.py --dataset $dataset" "$acc_dir" "$pool_dir" exp_table[@]

    echo "\n"
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

