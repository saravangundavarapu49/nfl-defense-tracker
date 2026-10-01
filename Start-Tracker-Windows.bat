@echo off
title NFL Defense Tracker
cd /d "%~dp0"

set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY goto nopython
%PY% -c "import sys; sys.exit(sys.version_info < (3, 10))" || goto oldpython

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting things up. This can take a few minutes...
    %PY% -m venv .venv || goto fail
)
if not exist ".venv\installed.txt" (
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto fail
    echo ok> ".venv\installed.txt"
)

echo Starting NFL Defense Tracker. Keep this window open while you use the app.
".venv\Scripts\python.exe" -m nfl_defense_tracker || goto fail
exit /b 0

:nopython
echo.
echo Python is not installed.
echo 1. Download it from the page that just opened: https://www.python.org/downloads/
echo 2. In the installer, tick "Add python.exe to PATH", then click Install Now.
echo 3. Double-click this file again.
start "" https://www.python.org/downloads/
pause
exit /b 1

:oldpython
echo.
echo Your Python is too old - version 3.10 or newer is needed.
echo Install the latest one from https://www.python.org/downloads/ and run this file again.
start "" https://www.python.org/downloads/
pause
exit /b 1

:fail
echo.
echo Something went wrong. Take a screenshot of this window and send it to Devin.
pause
exit /b 1
