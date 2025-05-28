@echo off
REM Run the entire processing pipeline.
cd /d "%~dp0.."
python Run_complete_program.py %*
