@echo off
REM Graceful stop request. The orchestrator terminates its collector trees.
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"
if not exist "data\staging\long_social_session" md "data\staging\long_social_session"
type nul > "data\staging\long_social_session\STOP"
echo Stop requested. The active worker and Reddit collector will be terminated.
echo Monitor shutdown with:
echo   Get-Content .\data\staging\long_social_session\logs\session.log -Wait -Tail 30
endlocal
