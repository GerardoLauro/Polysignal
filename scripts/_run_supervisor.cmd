@echo off
REM Launcher for the resilient social-session supervisor. Invoked by the Windows
REM Task Scheduler so the run is owned by the scheduler service (survives shells).
setlocal
cd /d "%~dp0.."
set "HOURS=%~1"
if "%HOURS%"=="" set "HOURS=10"
set "PYTHONUNBUFFERED=1"
".venv\Scripts\python.exe" "scripts\supervise_social_session.py" --total-hours %HOURS%
endlocal
