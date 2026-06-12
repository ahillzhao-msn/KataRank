<# 
.SYNOPSIS
    Install KataRank API Server as a Windows service using NSSM.

.DESCRIPTION
    This script registers katarank-server as a Windows service so it starts
    automatically on boot and runs in the background.

    Prerequisites:
      1. NSSM (Non-Sucking Service Manager) — download from:
         https://nssm.cc/download
         Place nssm.exe in PATH or in this scripts/ directory.

      2. Complete setup:
         uv sync --extra api
         # Verify katago binary is detected:
         katarank-server --model path\to\kata1.bin.gz --check-health

    Usage:
      # Install service
      .\scripts\install-service.ps1 -Model "C:\models\kata1-b18c384nbt.bin.gz"
      
      # With full options
      .\scripts\install-service.ps1 `
          -Model "C:\models\kata1.bin.gz" `
          -Checkpoint "C:\models\katarank\best.pt" `
          -Port 8765 `
          -SgfRoot "C:\data\sgf" `
          -KataGoBin "C:\katago\katago.exe"
      
      # Start the service
      net start KataRank
      
      # Check status
      nssm status KataRank
      
      # Stop
      net stop KataRank
      
      # Remove
      nssm remove KataRank confirm
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Model,
    
    [string]$Checkpoint,
    [string]$KataGoBin,
    [string]$Config,
    [string]$HumanModel,
    [string]$Host = "127.0.0.1",
    [int]$Port = 8765,
    [string]$SgfRoot,
    [int]$MaxConcurrency = 1,
    [ValidateSet("lite", "full")]
    [string]$EngineMode = "lite",
    [switch]$NoPersistent
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Nssm = Get-Command "nssm.exe" -ErrorAction SilentlyContinue
if (-not $Nssm) {
    $NssmLocal = Join-Path $PSScriptRoot "nssm.exe"
    if (Test-Path $NssmLocal) {
        $Nssm = $NssmLocal
    } else {
        Write-Error "NSSM not found. Download from https://nssm.cc/download and place nssm.exe in PATH or scripts/"
        exit 1
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Error ".venv not found. Run 'uv sync --extra api' first."
    exit 1
}

# Build argument list
$Args = @(
    "--model", $Model
    "--host", $Host
    "--port", $Port.ToString()
    "--max-concurrency", $MaxConcurrency.ToString()
    "--engine-mode", $EngineMode
)
if ($Checkpoint)   { $Args += "--checkpoint"; $Args += $Checkpoint }
if ($KataGoBin)    { $Args += "--katago-bin"; $Args += $KataGoBin }
if ($Config)       { $Args += "--config"; $Args += $Config }
if ($HumanModel)   { $Args += "--human-model"; $Args += $HumanModel }
if ($SgfRoot)      { $Args += "--sgf-root"; $Args += $SgfRoot }
if ($NoPersistent) { $Args += "--no-persistent" }

$AppPath = Join-Path $ProjectRoot ".venv\Scripts\katarank-server.exe"
if (-not (Test-Path $AppPath)) {
    Write-Warning "katarank-server.exe not found; using python -m katarank.api.server"
    $AppPath = $VenvPython
    $Args = @("-m", "katarank.api.server") + $Args
}

# Ensure log directory exists before NSSM tries to write to it
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

& $Nssm install KataRank $AppPath $Args
if ($LASTEXITCODE -ne 0) {
    Write-Error "NSSM install failed (exit code $LASTEXITCODE)"
    exit 1
}

# Configure service
& $Nssm set KataRank AppDirectory $ProjectRoot
& $Nssm set KataRank Description "KataRank — Go player rank assessment via KataGo neural analysis"
& $Nssm set KataRank Start SERVICE_AUTO_START
& $Nssm set KataRank AppStdout (Join-Path $ProjectRoot "logs\katarank-stdout.log")
& $Nssm set KataRank AppStderr (Join-Path $ProjectRoot "logs\katarank-stderr.log")
& $Nssm set KataRank AppRotateFiles 1
& $Nssm set KataRank AppRotateSeconds 86400

Write-Host ""
Write-Host "KataRank service installed. Start with:" -ForegroundColor Green
Write-Host "  net start KataRank" -ForegroundColor Cyan
Write-Host ""
Write-Host "View logs:" -ForegroundColor Green
Write-Host "  type logs\katarank-stdout.log" -ForegroundColor Cyan
Write-Host ""
Write-Host "Stop:" -ForegroundColor Green
Write-Host "  net stop KataRank" -ForegroundColor Cyan
Write-Host ""
Write-Host "Remove:" -ForegroundColor Green
Write-Host "  nssm remove KataRank confirm" -ForegroundColor Cyan
