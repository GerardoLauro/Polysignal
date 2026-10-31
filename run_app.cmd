@echo off
REM PolySignal — launch the full app (UI + live API) with one command.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [PolySignal] Could not find .venv\Scripts\python.exe
  echo            Run from the project root with the project virtualenv present.
  exit /b 1
)

echo [PolySignal] Starting local app server...
".venv\Scripts\python.exe" "scripts\serve_app.py" %*
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
