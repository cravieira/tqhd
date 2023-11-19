# Script for common variables and routines.

MODEL="./_model"

# Python environment management
function enable_venv() {
    source _venv/bin/activate
}

function disable_venv() {
    deactivate
}

function py_launch() {
    python3 $@
}
export -f py_launch

# HDC experiments

# Each experiment has its own table with the parameters used in the HDC
# pipeline. The entries are: vsa type, encode data type, associative memory
# data type, and dimension. The function parse_parameters is used to create the
# appropriate script arguments from a table entry.
voicehd_hdc_exp=(
    'map f32 f32 1000'
    'map f32 f32 10000'
    'map i32 i32 1000'
    'map i32 i32 10000'
    'map i16 i32 1000'
    'map i16 i32 10000'
    'bsc bool bool 1000'
    'bsc bool bool 10000'
    )

emg_hdc_exp=(
    'map f32 f32 1000'
    'map f32 f32 10000'
    )

mnist_hdc_exp=(
    'map f32 f32 1000'
    'map f32 f32 10000'
    )

language_hdc_exp=(
    'map f32 f32 1000'
    'map f32 f32 10000'
    )

# Create the command line used in HDC script models.
# $1: vsa type.
# $2: data type used by encoding.
# $3: data type used by associative memory.
# $4: vector size.
function parse_parameters() {
    local vsa=$1
    local enc=$2
    local am=$3
    local dim=$4
    # Using ${vsa^^} to invert it to upper case
    echo "--vsa ${vsa^^} --dtype-enc $enc --dtype-am $am --vector-size $dim"
}
