Get-Process -Id 31384 -ErrorAction SilentlyContinue | ForEach-Object { $_.Kill(); Write-Host "Killed pid $($_.Id)" }
# Also kill any orphaned katago processes
Get-Process -Name katago -ErrorAction SilentlyContinue | ForEach-Object { $_.Kill(); Write-Host "Killed katago pid $($_.Id)" }
Write-Host "Cleanup done"
