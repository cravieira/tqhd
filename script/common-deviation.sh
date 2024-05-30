# Quantize models using TQHD with different deviation values.
# This script trains models with dfferent random seeds and then quantize them
# using AM Thermometer choosing the quantization poles according to the
# deviation of the model. The models are experimented choosing a variable
# number of dimension expansion, i.e., 1 dimension MAP is expanded to different
# number of dimensions in the quantized vector. The number of quantization
# intervals is always the maximal number possible, i.e., <bits>+1. The purpose
# of this experiment is to understand how the accuracy of quantized models vary
# with different quantization values based on the standard deviation.
#
# This is the common deviation experiment implementation and shall not be
# executed. Other scripts call this script with proper parameters to execute
# the experiments accordingly.

set -eu

source common.sh

RESULT_DIR=_transformation # Result dir to be created
POOL_DIR=_pool # Pool dir to be created
JOBS=12 # Number of parallel jobs to be executed
DEVICE=cuda # Device used
# Name of the experiment executed by this script. Can be overriden by child
# scripts for custom behavior
EXP_NAME='deviation'

# DIMENSIONS: Space separated list of dimensions values to be experimented.
# Scripts that include this script can override this variable.
start=1000
step=1000
final=1000
DIMENSIONS=$(seq $start $step $final)

# Flag to dispatch experiments using vector normalization optimization. Child
# scripts must set this variable to activate the cache behavior.
TQHD_CACHE_NORM=''

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
#$3: Pointer to the parameter table
#$4 (Optional): Path to pool dir
function launch_hdc_table() {
    local cmd=$1
    local acc_dir=$2
    declare -a table=("${!3}")
    local pool_dir=""
    if [[ $# -eq 4 ]]; then
        pool_dir=$4
    fi

    for (( seed = 0; seed < $MAX_SEED; seed++ )); do
        for dim in $DIMENSIONS ; do
            for dev in $std_deviation ; do
                for (( i=0; i < ${#table[@]}; i++ )) ; do
                    vsa='--vsa MAP'
                    am_type='--am-type TQHD'
                    # Split the experiment table into its two variables
                    IFS=' ' read -r bits _dim_unused <<< ${table[i]}
                    local exp=$(parse_parameters $bits $dim)
                    local model_name="amtd-std$dev-bits$bits-d$dim"
                    local acc_file="$acc_dir/$model_name/$seed.acc"

                    # Create load command if there is a pool for this experiment
                    local load_cmd=""
                    if [ $pool_dir ]; then
                        load_cmd=$(create_load_cmd $pool_dir $dim $seed)
                    fi

                    local tqhd_cache_cmd=''
                    if [ $TQHD_CACHE_NORM ]; then
                        tqhd_cache_cmd='--am-tqhd-cache-norm'
                    fi

                    echo py_launch "$cmd $tqhd_cache_cmd $load_cmd $exp $vsa $am_type --am-tqhd-deviation $dev --device $DEVICE --seed $seed --accuracy-file $acc_file"
                done
            done
        done
    done

}

function voicehd() {
    local app="voicehd"
    local acc_dir="$RESULT_DIR/$app/hdc/$EXP_NAME"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch_hdc_table "src/voicehd_hdc.py" "$acc_dir" exp_table[@]  "$pool_dir"
    echo "\n"
}

function emg() {
    local app="emg-all"
    local acc_dir="$RESULT_DIR/$app/hdc/$EXP_NAME"
    launch_hdc_table "src/emg.py" "$acc_dir" exp_table[@]
    echo "\n"
}

function mnist() {
    local app="mnist"
    local acc_dir="$RESULT_DIR/$app/hdc/$EXP_NAME"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch_hdc_table "src/mnist_hdc.py " "$acc_dir" exp_table[@] "$pool_dir"
    echo "\n"
}

function language() {
    local app="language"
    local acc_dir="$RESULT_DIR/$app/hdc/$EXP_NAME"
    local pool_dir="$POOL_DIR/$app/hdc"
    launch_hdc_table "src/language.py " "$acc_dir" exp_table[@] "$pool_dir"
    echo "\n"
}

function hdchog() {
    local app="hdchog-fashionmnist"
    local acc_dir="$RESULT_DIR/$app/hdc/$EXP_NAME"
    local pool_dir="$POOL_DIR/$app/hdc"
    local dataset="FashionMNIST"

    launch_hdc_table "src/hdchog.py --dataset $dataset" "$acc_dir" exp_table[@] "$pool_dir"

    echo "\n"
}

# Launch GraphHD experiment for a given dataset.
# $1: Name of the dataset. Must be one of the dataset options avaible in
#   graphhd.py
function graphhd_dataset() {
    local dataset="$1"
    local lower_case_ds=$(com_to_lowercase "$dataset")
    local app="graphhd-$lower_case_ds"
    local acc_dir="$RESULT_DIR/$app/hdc/$EXP_NAME"
    local pool_dir="$POOL_DIR/$app/hdc"

    launch_hdc_table "src/graphhd.py --dataset $dataset" "$acc_dir" exp_table[@] "$pool_dir"

    echo "\n"
}
