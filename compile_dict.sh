#!/usr/bin/env bash
# Quick script to re-compile all text dictionaries in Data/my dataset to binary formats
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 "${SCRIPT_DIR}/compile_dictionaries.py"
