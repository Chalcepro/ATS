@echo off
rem Double-click to train. Runs headless in this window and never stops on its
rem own: if training ends or crashes, it says why and starts again 15 seconds
rem later, carrying on from the saved brain. Close the window to stop it.
rem
rem Refuses to start a second copy: two runs writing brain.pt at once would
rem corrupt it.
title ATS training
cd /d "%~dp0"

set "PY=python"
%PY% -c "import torch" >nul 2>&1
if errorlevel 1 set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'main\.py' -and $_.CommandLine -match 'no-gui' }) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo.
    echo   Training is already running in another window or in the background.
    echo   Stop that one first - two runs at once would damage brain.pt.
    echo.
    pause
    exit /b 1
)

:loop
echo.
echo ==== training started %date% %time% ====
"%PY%" main.py --no-gui
echo.
echo ==== training stopped %date% %time% (exit code %errorlevel%) ====
echo      restarting in 15 seconds - close this window to stop for good
timeout /t 15
goto loop
