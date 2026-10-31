@echo off
REM ===========================================================================
REM PolySignal - resumable Scrapling public-web enrichment (Reddit + Telegram).
REM
REM Writes to data\staging\{reddit,telegram_web,enriched} ONLY (never canonical
REM data\normalized). Safe to kill and re-run: every fetched page is flushed and
REM checkpointed before its cursor advances. A live-PID lock
REM (data\staging\enriched\scrapling_web.lock) refuses to start a duplicate.
REM
REM Reddit on old.reddit.com routinely returns HTTP 403/429 to automated fetches.
REM The collector OPENS A CIRCUIT BREAKER, reports the block in the run summary,
REM and respects it (6h cooldown). It does NOT bypass or evade it. Telegram
REM public-web (t.me/s/<channel>) collection continues regardless.
REM
REM Usage:   scripts\collect_scrapling_web.cmd [TARGET_TOTAL_RECORDS]
REM Default: TARGET_TOTAL_RECORDS = 400000
REM ===========================================================================
setlocal
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=400000"
set "OUT=data\staging\enriched"
set "LOGDIR=%OUT%\reports"
set "LOG=%LOGDIR%\scrapling_web_collect.log"

cd /d "%PROJECT_ROOT%"
if errorlevel 1 ( echo ERROR: cannot cd to "%PROJECT_ROOT%" & exit /b 1 )

if not exist ".venv\Scripts\activate.bat" (
  echo ERROR: virtualenv not found at "%PROJECT_ROOT%\.venv". Create it first.
  exit /b 1
)
call .venv\Scripts\activate.bat
set "PYTHONUNBUFFERED=1"

REM Create output + log directories BEFORE the redirect opens the log file.
if not exist "data\staging\reddit" md "data\staging\reddit"
if not exist "data\staging\telegram_web" md "data\staging\telegram_web"
if not exist "%LOGDIR%" md "%LOGDIR%"

echo Starting Scrapling public-web collection -> "%LOG%"  (target=%TARGET%)
echo [%date% %time%] START collect-large-enriched target-total-records=%TARGET%>> "%LOG%"
python -m polysignal social collect-large-enriched ^
  --config configs\social_enrichment_sources.example.yaml ^
  --markets-path data\normalized\polymarket\markets.parquet ^
  --reddit-root data\staging\reddit ^
  --telegram-root data\staging\telegram_web ^
  --output-root "%OUT%" ^
  --target-total-records %TARGET% ^
  --target-reddit-records 250000 ^
  --target-telegram-records 150000 ^
  --max-pages-per-source 200 ^
  --max-pages-per-query 5 ^
  --max-results-per-query 100 ^
  --rate-limit-seconds 5.0 ^
  --resume >> "%LOG%" 2>&1
set "RC=%errorlevel%"
echo [%date% %time%] DONE  collect-large-enriched exit=%RC%>> "%LOG%"
if "%RC%"=="3" echo NOTE: another Scrapling collector already holds the lock - not started.
if not "%RC%"=="0" if not "%RC%"=="3" echo Scrapling collector exited with code %RC% - see "%LOG%".
endlocal
