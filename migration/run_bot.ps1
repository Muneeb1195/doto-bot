$RepoDir = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoDir ".venv\Scripts\pythonw.exe"
$Script = Join-Path $RepoDir "bot\main.py"
$LogDir = Join-Path $RepoDir "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$LogFile = Join-Path $LogDir "bot_$(Get-Date -Format 'yyyyMMdd').log"
$PyOut = Join-Path $LogDir "bot_py.out"
$PyErr = Join-Path $LogDir "bot_py.err"

while ($true) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    try {
        $p = Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList $Script -PassThru `
            -RedirectStandardOutput $PyOut -RedirectStandardError $PyErr
        $p.WaitForExit()
    } catch {
        Add-Content -Path $LogFile -Value "$timestamp [FATAL] Bot process crashed: $_"
    }
    Add-Content -Path $LogFile -Value "$timestamp [INFO] Bot exited, restarting in 5s..."
    Start-Sleep -Seconds 5
}
