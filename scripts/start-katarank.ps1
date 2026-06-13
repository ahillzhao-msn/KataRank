<#
.SYNOPSIS
    Start the KataRank API server in the background (if not already running).

.DESCRIPTION
    Designed to be called from WSL/GoPredict via Windows interop:
        powershell.exe -NonInteractive -File "C:\path\to\katarank\scripts\start-katarank.ps1"

    - If the server is already up (health check passes), exits immediately.
    - Otherwise launches katarank-server as a background job with logs
      written to logs\katarank-stdout.log / katarank-stderr.log.
    - All settings come from ~/.katarank/server.toml — no args needed.
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Port = 8765   # must match server.toml [katarank] port

# ── Already running? ──────────────────────────────────────────────────────────
try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" `
                              -TimeoutSec 3 -ErrorAction Stop
    if ($resp.status -eq "ok" -or $resp.status -eq "degraded") {
        Write-Host "KataRank already running (status: $($resp.status))"
        exit 0
    }
} catch {
    # Not up — continue to start it
}

# ── Launch ────────────────────────────────────────────────────────────────────
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Stdout = Join-Path $LogDir "katarank-stdout.log"
$Stderr = Join-Path $LogDir "katarank-stderr.log"
$Launcher = Join-Path $ProjectRoot ".venv\Scripts\katarank-server.exe"

if (-not (Test-Path $Launcher)) {
    $Launcher = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $LaunchArgs = "-m katarank.api.server"
} else {
    $LaunchArgs = ""
}

if (-not (Test-Path $Launcher)) {
    Write-Error "katarank not installed. Run: uv sync --extra api"
    exit 1
}

Write-Host "Starting KataRank server..."
$proc = Start-Process -FilePath $Launcher `
    -ArgumentList $LaunchArgs `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError  $Stderr `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Launched (pid $($proc.Id)) — waiting for health check..."

# ── Wait up to 60 s for the server to be ready ────────────────────────────────
$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" `
                                  -TimeoutSec 3 -ErrorAction Stop
        Write-Host "KataRank ready (status: $($resp.status))"
        exit 0
    } catch { }

    if ($proc.HasExited) {
        Write-Error "KataRank process exited unexpectedly. Check: $Stderr"
        exit 1
    }
}

Write-Error "KataRank did not become ready within 60 s. Check: $Stderr"
exit 1
