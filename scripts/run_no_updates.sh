#!/bin/bash
# Skip database updates and directly calculate indicators and ranking.
# Equivalent to: python Run_complete_program.py --skip-update-price-data --skip-update-finance-data
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir/.."  # change to repository root
python3 Run_complete_program.py --skip-update-price-data --skip-update-finance-data "$@"
