#!/bin/bash
# Update price data only, skipping finance data update.
# Equivalent to: python Run_complete_program.py --skip-update-finance-data
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir/.."
python3 Run_complete_program.py --skip-update-finance-data "$@"
