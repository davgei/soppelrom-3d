@echo off
rem Start the control panel. Same reason as nettleser.cmd: a bare `python` is the system Python and
rem has none of the project's packages, so it fails one import at a time.
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Fant ikke .venv\Scripts\python.exe i %~dp0
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m src.dashboard %*
if errorlevel 1 pause
