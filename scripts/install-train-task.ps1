<#
.SYNOPSIS
    Register a Windows Scheduled Task to check for KataRank training daily.

.DESCRIPTION
    Creates a scheduled task "KataRank-AutoTrain" that runs check-train.ps1
    daily at the specified time. The script checks the GoPredict DB game
    count and triggers training if the threshold is met.

.EXAMPLE
    # Install with defaults (daily at 3:00 AM)
    .\scripts\install-train-task.ps1

    # Custom time
    .\scripts\install-train-task.ps1 -Time "06:00"

    # Remove the task
    Unregister-ScheduledTask -TaskName "KataRank-AutoTrain" -Confirm:$false
#>

param(
    [string]$Time = "03:00",
    [string]$TaskName = "KataRank-AutoTrain"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CheckScript = Join-Path $ProjectRoot "scripts\check-train.ps1"

if (-not (Test-Path $CheckScript)) {
    Write-Error "check-train.ps1 not found at $CheckScript"
    exit 1
}

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task '$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File `"$CheckScript`"" `
    -WorkingDirectory $ProjectRoot

$trigger = New-ScheduledTaskTrigger -Daily -At $Time

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Check GoPredict DB and trigger KataRank model training every 3000 new games"

Write-Host ""
Write-Host "Scheduled task '$TaskName' created." -ForegroundColor Green
Write-Host "  Schedule: Daily at $Time" -ForegroundColor Cyan
Write-Host "  Script:   $CheckScript" -ForegroundColor Cyan
Write-Host ""
Write-Host "Management:" -ForegroundColor Green
Write-Host "  Run now:  Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Cyan
Write-Host "  Status:   Get-ScheduledTask -TaskName '$TaskName' | Select State" -ForegroundColor Cyan
Write-Host "  Remove:   Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" -ForegroundColor Cyan
