# Script for common variables and routines.

# Common variables #
# GraphHD datasets evaluated
GRAPHHD_DATASETS=("DD" "ENZYMES" "MUTAG" "NCI1" "PROTEINS" "PTC_FM")

# Python environment management #
function enable_venv() {
    source _venv/bin/activate
}

function disable_venv() {
    deactivate
}

function py_launch() {
    python3 $@
}
# Export py_launch so GNU parallel can call it.
export -f py_launch

# Common used functions #

# Execute the function in $1 for all values in the array given in $2. Extra
# arguments are passed down the hierarchy.
# $1: A reference to the function to be applied.
# $2: A reference to an array containing the name of the datasets.
# $@ (Optional/Any number of variables): Other arguments that should be passed
#   to the function reference in $2.
function com_foreach() {
    local func_name="$1"
    shift
    local -n datasets="$1"
    shift

    for dataset in "${datasets[@]}"; do
        $func_name "$dataset" "$@"
    done
}

# Transform a string to lower case
# $1: String.
function com_to_lowercase() {
    local str="$1"
    echo $str | tr '[:upper:]' '[:lower:]'
}

# Check if a function name exists.
# $1: Function name.
function com_fn_exists() {
    local fname="$1"
    declare -f "$fname" > /dev/null && echo '1'
}

## Functions to launch programs

# Since GraphHD has several datasets available, this function is responsible
# for evaluating all datasets defined in GRAPHHD_DATASETS. This function calls
# a function named "graphhd_dataset", which in turn call the experiment
# launcher and sets up the proper command. Thus, each experiment script file
# *must* define its own "graphhd_dataset" function.
function com_graphhd() {
    local fname='graphhd_dataset'
    if [ ! $(com_fn_exists "$fname") ]; then
        echo "Function \"$fname\" not defined."
        exit 1
    fi

    # Ensure CPU usage in this app since CUDA might consume a lot of GPU RAM
    local old_device="$DEVICE"
    DEVICE='cpu'
    com_foreach "$fname" "GRAPHHD_DATASETS"
    # Restore previous device used
    DEVICE="$old_device"

    echo "\n"
}

# Launch a batch of jobs in parallel using GNU parallel.
# If any of the jobs fail, then all other jobs are immediately killed.
# $1: Number of simultaneous jobs
# $2: A string containing the commands to be launched separated by new line
function parallel_launch() {
    local jobs=$1
    local cmds="$2"

    # File to control the number of simultaneous jobs in gnu parallel. This bash
    # script launches a different number of jobs depending on the HDC scripts
    # running. This fine-grained control is useful to execute more jobs in
    # parallel in applications that are not so hardware demanding.
    PROCFILE=$(mktemp -p . _tqhd-procfile.XXX)
    echo "Using \"$PROCFILE\" as procfile for job control in GNU parallel..."

    # Clean up temp file if script fails
    trap "rm $PROCFILE" QUIT TERM PWR EXIT

    echo "$jobs" > $PROCFILE
    printf "$cmds" | parallel --verbose -j"$PROCFILE" --halt now,fail=1
}
