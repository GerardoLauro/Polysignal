@echo off
REM Ten-hour Reddit + X posts/comments session. Runs in the current terminal.
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "HOURS=%~1"
if "%HOURS%"=="" set "HOURS=10"

cd /d "%PROJECT_ROOT%"
if errorlevel 1 ( echo ERROR: cannot cd to "%PROJECT_ROOT%" & exit /b 1 )
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: virtualenv not found at "%PROJECT_ROOT%\.venv".
  exit /b 1
)

set "PYTHONUNBUFFERED=1"
echo Starting %HOURS%-hour Reddit + X posts/comments session.
echo Session log: data\staging\long_social_session\logs\session.log
".venv\Scripts\python.exe" scripts\run_long_social_session.py ^
  --hours %HOURS% ^
  --cycle-minutes 30 ^
  --reddit-target 500000 ^
  --x-target-per-pass 6000 ^
  --x-accounts-per-pass 8 ^
  --reddit-pass-minutes 12 ^
  --reddit-comment-posts-per-pass 50 ^
  --x-comment-posts-per-pass 25 ^
  --comments-per-post 10 ^
  --reddit-rate-limit 3 ^
  --comment-rate-limit 3
set "RC=%errorlevel%"
echo Long session exit=%RC%
endlocal & exit /b %RC%
