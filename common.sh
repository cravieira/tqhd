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
