$ProjectRoot = "C:\Users\xiaoj\KataRank"
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Stdout = Join-Path $LogDir "katarank-stdout.log"
$Stderr = Join-Path $LogDir "katarank-stderr.log"

# Kill any previous instances
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -match "katarank"
} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 1

$proc = Start-Process -FilePath (Join-Path $ProjectRoot ".venv\Scripts\python.exe") `
    -ArgumentList "-m katarank.api.server" `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden -PassThru

Write-Host "Launched (pid: $($proc.Id))"

# Wait and check
Start-Sleep -Seconds 8
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "HEALTH: status=$($r.status)"
} catch {
    Write-Host "HEALTH FAILED: $_"
    Write-Host "=== STDERR (last 20 lines) ==="
    Get-Content $Stderr -Tail 20 -ErrorAction SilentlyContinue
    Write-Host "=== STDOUT (last 20 lines) ==="
    Get-Content $Stdout -Tail 20 -ErrorAction SilentlyContinue
}
