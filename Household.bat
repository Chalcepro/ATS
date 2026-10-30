@echo off
rem One place for the household world - no commands to remember.
cd /d "%~dp0"
set PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
:menu
echo.
echo   1  Talk  - watch the house hour by hour and type to it
echo   2  Exam  - how the trained brain does
echo   3  Train - teach the brain again (about 20 minutes)
echo   4  Quit
set /p c=Pick: 
if "%c%"=="1" "%PY%" -m household.talk
if "%c%"=="2" "%PY%" -m household.exam
if "%c%"=="3" "%PY%" -m household.train
if "%c%"=="4" exit /b
goto menu
