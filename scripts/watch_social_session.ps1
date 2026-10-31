param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "SilentlyContinue"
$logRoot = Join-Path $ProjectRoot "data\staging\long_social_session\logs"
$statePath = Join-Path $ProjectRoot "data\staging\long_social_session\session_state.json"
$files = @(
    @{ Tag = "SESSION"; Path = (Join-Path $logRoot "session.log") },
    @{ Tag = "REDDIT"; Path = (Join-Path $logRoot "reddit_posts.log") },
    @{ Tag = "X"; Path = (Join-Path $logRoot "x_posts.log") },
    @{ Tag = "COMMENTS"; Path = (Join-Path $logRoot "comments.log") }
)

Write-Host "PolySignal live logs. Ctrl+C stops this viewer only." -ForegroundColor Cyan
Write-Host "Use scripts\stop_long_social_session.cmd to stop the scraping session." -ForegroundColor Yellow

$deadline = (Get-Date).AddSeconds(30)
while (-not (Test-Path $statePath) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
}

$jobs = @()
foreach ($file in $files) {
    $path = $file.Path
    $tag = $file.Tag
    while (-not (Test-Path $path) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 250
    }
    if (Test-Path $path) {
        $jobs += Start-Job -ArgumentList $path, $tag -ScriptBlock {
            param($LogPath, $Tag)
            Get-Content -LiteralPath $LogPath -Wait -Tail 20 |
                ForEach-Object { "[$Tag] $_" }
        }
    }
}

try {
    $completedChecks = 0
    while ($true) {
        foreach ($job in $jobs) {
            Receive-Job -Job $job
        }
        if (Test-Path $statePath) {
            try {
                $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
                if ($state.status -eq "completed") {
                    $completedChecks += 1
                } else {
                    $completedChecks = 0
                }
            } catch {
                $completedChecks = 0
            }
        }
        if ($completedChecks -ge 6) {
            Write-Host "Session completed. Live viewer exiting." -ForegroundColor Green
            break
        }
        Start-Sleep -Milliseconds 500
    }
} finally {
    $jobs | Stop-Job
    $jobs | Remove-Job
}
