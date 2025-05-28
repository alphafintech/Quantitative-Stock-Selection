@echo off
REM Skip updates and directly calculate indicators and ranking.
cd /d "%~dp0.."

set "PYTHONUTF8=1"
python -X utf8 Run_complete_program.py --skip-update-price-data --skip-update-finance-data %*
