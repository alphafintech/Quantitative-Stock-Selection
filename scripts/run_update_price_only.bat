@echo off
REM Update price data only, skipping finance data update.
cd /d "%~dp0.."
python Run_complete_program.py --skip-update-finance-data %*
