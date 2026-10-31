@echo off
REM One-command stable 10-hour Reddit + X session with multiplexed live logs.
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

echo Stopping old scraper/session processes and clearing stale locks...
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\force_stop_social_session.ps1"
if exist "data\staging\long_social_session\session_state.json" del /q "data\staging\long_social_session\session_state.json"
if exist "data\staging\long_social_session\STOP" del /q "data\staging\long_social_session\STOP"

set "PYTHONUNBUFFERED=1"
echo Starting stable %HOURS%-hour session in the background...
powershell -NoProfile -Command ^
  "$args=@('scripts\run_long_social_session.py','--hours','%HOURS%','--cycle-minutes','30','--reddit-target','500000','--x-target-per-pass','6000','--x-accounts-per-pass','8','--reddit-pass-minutes','12','--reddit-comment-posts-per-pass','50','--x-comment-posts-per-pass','25','--comments-per-post','10','--reddit-rate-limit','3','--comment-rate-limit','3'); $p=Start-Process -FilePath '%PROJECT_ROOT%\.venv\Scripts\python.exe' -ArgumentList $args -WorkingDirectory '%PROJECT_ROOT%' -WindowStyle Hidden -PassThru; New-Item -ItemType Directory -Force -Path '%PROJECT_ROOT%\data\staging\long_social_session' | Out-Null; Set-Content -Path '%PROJECT_ROOT%\data\staging\long_social_session\launcher_pid.txt' -Value $p.Id; Write-Host ('Worker PID: ' + $p.Id)"

echo Waiting for session logs...
ping 127.0.0.1 -n 4 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%\scripts\watch_social_session.ps1" -ProjectRoot "%PROJECT_ROOT%"
endlocal
