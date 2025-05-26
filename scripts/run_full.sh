#!/bin/bash
# Run the entire processing pipeline.
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir/.."
python3 Run_complete_program.py "$@"
