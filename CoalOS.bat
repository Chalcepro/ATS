@echo off
rem CoalOS - Coal's pretend computer. Opens in your browser; close this window to shut it down.
cd /d "%~dp0"
start "" http://127.0.0.1:8766
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m coalos.server
