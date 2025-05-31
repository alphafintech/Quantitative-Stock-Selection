@echo off
REM Skip updates and directly calculate indicators and ranking.
cd /d "%~dp0.."

set "PYTHONUTF8=1"
python -X utf8 post_process.py  %*
