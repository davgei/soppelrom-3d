@echo off
rem Start the browser view. Double-click this, or run it from any terminal.
rem
rem Why a .cmd instead of "python -m src.web": a bare `python` on this machine is the system
rem Python 3.12, which has none of the project's packages -- so the command fails on numpy, then on
rem flask, then on open3d, one import at a time. Everything lives in .venv, and this points there
rem explicitly so the working directory and the active shell cannot change which Python runs.
setlocal
rem Python writes UTF-8 (-X utf8) but the console starts in cp850, which turns "kjør «Generer»" into
rem "kj├╕r ┬½Generer┬╗". >nul keeps chcp's own chatter out of the way.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Fant ikke .venv\Scripts\python.exe i %~dp0
    echo Kjor:  python -m venv .venv  og deretter  .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m src.web %*
if errorlevel 1 pause
