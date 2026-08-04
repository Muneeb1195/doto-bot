if (-not $env:DASHBOARD_USER) { $env:DASHBOARD_USER = [Environment]::GetEnvironmentVariable("DASHBOARD_USER", "User") }
if (-not $env:DASHBOARD_PASS) { $env:DASHBOARD_PASS = [Environment]::GetEnvironmentVariable("DASHBOARD_PASS", "User") }
if (-not $env:DASHBOARD_USER) { $env:DASHBOARD_USER = "admin" }
if (-not $env:DASHBOARD_PASS) { $env:DASHBOARD_PASS = "dNBQJlzbPuVIwWSC" }
$RepoDir = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoDir ".venv\Scripts\pythonw.exe"
$LogDir = Join-Path $RepoDir "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$LogFile = Join-Path $LogDir "dashboard.log"
$PyOut = Join-Path $LogDir "dashboard_py.out"
$PyErr = Join-Path $LogDir "dashboard_py.err"

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "$timestamp [INFO] Starting dashboard (DASHBOARD_USER='$env:DASHBOARD_USER')..."
    $p = Start-Process -WindowStyle Hidden -FilePath $Python -WorkingDirectory $RepoDir `
        -ArgumentList "-m","uvicorn","dashboard.api:app","--host","0.0.0.0","--port","8501" `
        -PassThru -RedirectStandardOutput $PyOut -RedirectStandardError $PyErr
    $p.WaitForExit()
    Add-Content -Path $LogFile -Value "$timestamp [INFO] Dashboard exited (code $($p.ExitCode)), restarting in 5s..."
    Start-Sleep -Seconds 5
}
