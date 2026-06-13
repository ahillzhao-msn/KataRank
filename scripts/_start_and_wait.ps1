$ProjectRoot = "C:\Users\xiaoj\KataRank"
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Stdout = Join-Path $LogDir "katarank-stdout.log"
$Stderr = Join-Path $LogDir "katarank-stderr.log"

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Write-Host "Starting: $Python -m katarank.api.server"

# Load .env manually so dotenv doesn't need to be configured
$env:PYTHONUNBUFFERED = "1"

$proc = Start-Process -FilePath $Python `
    -ArgumentList "-m katarank.api.server" `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden -PassThru

Write-Host "Launched (pid: $($proc.Id))"

# Wait for readiness — poll health endpoint
$deadline = (Get-Date).AddSeconds(120)
$ready = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 5 -ErrorAction Stop
        Write-Host "READY: status=$($r.status)"
        $ready = $true
        break
    } catch {
        Write-Host "." -NoNewline
    }
    if ($proc.HasExited) {
        Write-Host ""
        Write-Host "PROCESS EXITED unexpectedly"
        Write-Host "=== STDERR (tail 30) ==="
        Get-Content $Stderr -Tail 30 -ErrorAction SilentlyContinue
        Write-Host "=== STDOUT (tail 30) ==="
        Get-Content $Stdout -Tail 30 -ErrorAction SilentlyContinue
        exit 1
    }
}

if (-not $ready) {
    Write-Host ""
    Write-Host "TIMEOUT: Not ready within 120s"
    Write-Host "=== STDERR ==="
    Get-Content $Stderr -Tail 50 -ErrorAction SilentlyContinue
    Write-Host "=== STDOUT ==="
    Get-Content $Stdout -Tail 50 -ErrorAction SilentlyContinue
    exit 1
}
