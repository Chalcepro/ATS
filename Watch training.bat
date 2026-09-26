@echo off
rem Double-click while a training run is going. It finds the run by itself.
title ATS training watch
cd /d "%~dp0"
set "PY=python"
%PY% -c "import tkinter" >nul 2>&1
if errorlevel 1 set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
start "" "%PY%" trainwatch.py %*
