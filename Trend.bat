@echo off
rem Double-click while training is running to watch the trend update live.
rem Reads the logs only - cannot slow training down.
title ATS trend
cd /d "%~dp0"
set "PY=python"
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
"%PY%" trend.py --watch 60
pause
