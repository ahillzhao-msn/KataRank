<#
.SYNOPSIS
    Periodic check: trigger KataRank training if enough new games exist.

.DESCRIPTION
    Designed to run as a Windows Scheduled Task (e.g., daily at 3 AM).
    Calls auto_train.py which checks the GoPredict DB game count
    against the last training threshold.

    Install as scheduled task:
        schtasks /create /tn "KataRank-AutoTrain" /tr "powershell -NonInteractive -File C:\Users\xiaoj\KataRank\scripts\check-train.ps1" /sc daily /st 03:00 /ru SYSTEM

    Or use the companion install-train-task.ps1 script.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "scripts\auto_train.py"
$LogDir = Join-Path $ProjectRoot "logs"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogFile = Join-Path $LogDir "auto_train_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

# Ensure WSL/Docker is running (PostgreSQL lives there)
wsl -d Debian -- bash -c 'docker ps -q' 2>$null | Out-Null

& $Python $Script 2>&1 | Tee-Object -FilePath $LogFile
