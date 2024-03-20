# Script for common variables and routines.

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

    echo "$JOBS" > $PROCFILE
    printf "$cmds" | parallel --verbose -j"$PROCFILE" --halt now,fail=1
}
