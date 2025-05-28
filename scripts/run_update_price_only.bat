@echo off
REM Update price data only, skipping finance data update.
cd /d "%~dp0.."

set "PYTHONUTF8=1"
python -X utf8 Run_complete_program.py --skip-update-finance-data %*
