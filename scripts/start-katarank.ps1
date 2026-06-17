<#
.SYNOPSIS
    Start/stop the KataRank API server headless (no terminal window).

.DESCRIPTION
    All settings come from ~/.katarank/server.toml.

    Usage:
        .\scripts\start-katarank.ps1            # start
        .\scripts\start-katarank.ps1 -Stop      # stop
        .\scripts\start-katarank.ps1 -Restart    # restart (reload new code)
#>

param(
    [switch]$Stop,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Port = 8765

function Test-ServerUp {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:${Port}/health" -TimeoutSec 3 -ErrorAction Stop
        return ($r.status -eq "ok" -or $r.status -eq "degraded")
    } catch {
        return $false
    }
}

function Stop-KataRank {
    $procs = Get-Process -Name "katarank-server" -ErrorAction SilentlyContinue
    if ($procs) {
        $procs | Stop-Process -Force -Confirm:$false
        Write-Host "Stopped katarank-server (pid $($procs.Id -join ', '))"
    } else {
        Write-Host "No katarank-server process found"
    }
    Start-Sleep -Seconds 2
}

if ($Stop) {
    Stop-KataRank
    exit 0
}

if ($Restart) {
    Write-Host "Restarting..."
    Stop-KataRank
}
elseif (Test-ServerUp) {
    Write-Host "KataRank already running on port $Port"
    exit 0
}

# ---- Find launcher ----
$ServerExe = Join-Path $ProjectRoot ".venv\Scripts\katarank-server.exe"
if (-not (Test-Path $ServerExe)) {
    Write-Error "katarank not installed. Run: uv sync --extra api"
    exit 1
}

# ---- Prepare logs ----
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$StdoutLog = Join-Path $LogDir "katarank-stdout.log"
$StderrLog = Join-Path $LogDir "katarank-stderr.log"

# ---- Launch headless ----
Write-Host "Starting KataRank server..."
$proc = Start-Process -FilePath $ServerExe `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError  $StderrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Launched (pid $($proc.Id)). Waiting for health check..."

# ---- Wait for ready ----
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 3
    if (Test-ServerUp) {
        Write-Host "KataRank ready on port $Port (pid $($proc.Id))" -ForegroundColor Green
        exit 0
    }
    if ($proc.HasExited) {
        Write-Host "Process exited (code $($proc.ExitCode)). Check: $StderrLog" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Server did not become ready within 90s. Check: $StderrLog" -ForegroundColor Yellow
exit 1
