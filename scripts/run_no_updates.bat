@echo off
REM Skip updates and directly calculate indicators and ranking.
cd /d "%~dp0.."
python Run_complete_program.py --skip-update-price-data --skip-update-finance-data %*
