#!/bin/bash

set -e

# Check if the computer runs Ubuntu 22.04 and download python3-venv.
if [[ $(lsb_release -r) == *"22.04"* ]]; then
    if [[ ! $(dpkg -s python3-venv) ]]; then
        echo "python3-venv not available. It is necessary sudo to install it:"
        sudo apt install -y python3-venv
    fi
else
    echo "Unsupported OS. Please install python3-venv"
    exit 1
fi

# Clone torchhd v5.0.1
git clone

# Create virtual environment and download packages
python3 -m venv _venv
source _venv/bin/activate
pip install -r requirements.txt
# Enable local torchhd in the virtual environment
pip install -e torchhd
deactivate
