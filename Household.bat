@echo off
rem One place for the household world - no commands to remember.
cd /d "%~dp0"
set PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
:menu
echo.
echo   0  Window - talk to it in the browser (mic, camera, voice if plugged in)
echo   1  Talk   - the same, in this black window
echo   2  Exam  - how the trained brain does
echo   3  Train - teach the brain again (about 20 minutes)
echo   4  Quit
set /p c=Pick: 
if "%c%"=="0" (start "" http://127.0.0.1:8765 & "%PY%" -m household.app)
if "%c%"=="1" "%PY%" -m household.talk
if "%c%"=="2" "%PY%" -m household.exam
if "%c%"=="3" "%PY%" -m household.train
if "%c%"=="4" exit /b
goto menu
