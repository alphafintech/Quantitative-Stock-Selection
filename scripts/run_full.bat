@echo off
REM Run the entire processing pipeline.
cd /d "%~dp0.."

set "PYTHONUTF8=1"
python -X utf8 Run_complete_program.py %*
