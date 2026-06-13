$ProjectRoot = "C:\Users\xiaoj\KataRank"
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Stdout = Join-Path $LogDir "katarank-stdout.log"
$Stderr = Join-Path $LogDir "katarank-stderr.log"

$proc = Start-Process -FilePath (Join-Path $ProjectRoot ".venv\Scripts\python.exe") `
    -ArgumentList "-m katarank.api.server" `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden -PassThru

Write-Host ("Launched (pid: " + $proc.Id + ")")
