try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 10 -ErrorAction Stop
    Write-Host "status=$($r.status)"
} catch {
    Write-Host "NOT READY: $_"
}
