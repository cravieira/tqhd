#!/bin/bash

# Create paper plots and results.

source common.sh

enable_venv
python3 src/paper-plots.py
disable_venv
