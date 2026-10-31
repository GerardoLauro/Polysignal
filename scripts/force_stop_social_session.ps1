param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

$patterns = @(
    "run_long_social_session.py",
    "collect_twitter_playwright.py",
    "collect_social_comments.py",
    "collect-large-enriched",
    "monitor_social_ingestion.py",
    "hourly_social_loop.py"
)

$processes = Get-CimInstance Win32_Process | Where-Object {
    $commandLine = $_.CommandLine
    $nameMatches = $_.Name -match "^(python|pythonw|cmd)\.exe$"
    $patternHits = @($patterns | Where-Object { $commandLine -like "*$_*" })
    $nameMatches -and
    $commandLine -and
    ($commandLine.IndexOf($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) -and
    $patternHits.Count -gt 0
}

foreach ($process in ($processes | Sort-Object ProcessId -Descending)) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

$staging = Join-Path $ProjectRoot "data\staging"
Get-ChildItem -LiteralPath $staging -Recurse -Filter *.lock -File -ErrorAction SilentlyContinue |
    ForEach-Object {
        $remove = $false
        try {
            $payload = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
            $holder = Get-Process -Id ([int]$payload.pid) -ErrorAction SilentlyContinue
            if (-not $holder) { $remove = $true }
        } catch {
            $remove = $true
        }
        if ($remove) {
            Remove-Item -LiteralPath $_.FullName -Force
        }
    }
