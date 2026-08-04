<# Redeploy the Doto MT5 bot + dashboard cleanly and verify they came back up.
#
# Uses the svc_launcher watchdog (.child.pid) to kill any orphaned previous
# child before/after the restart, then polls bot.log for the "Bot state loaded"
# marker so a broken deploy fails loudly instead of silently leaving the bot
# down (agent audit D2 / redeploy-on-audit loop).
#>
param(
    [string[]] $Tasks = @("DotoBot", "DotoDashboard"),
    [int] $PollSeconds = 60,
    [int] $PollInterval = 3
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Repo "logs"

function Kill-ChildPid {
    param([string] $Name)
    $pidFile = Join-Path $LogDir "$Name.child.pid"
    if (-not (Test-Path $pidFile)) { return }
    $pid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Out-String).Trim()
    if ($pid -match '^\d+$') {
        Write-Host "[redeploy] killing previous $Name child pid=$pid"
        & taskkill /PID $pid /T /F 2>$null | Out-Null
    }
}

function Wait-ForBotLoaded {
    $log = Join-Path $LogDir "bot.log"
    $deadline = (Get-Date).AddSeconds($PollSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path $log) {
            $tail = Get-Content $log -Tail 200 -ErrorAction SilentlyContinue
            if ($tail -match "Bot state loaded") {
                if ($tail -notmatch "Traceback|Error") {
                    Write-Host "[redeploy] bot.log shows clean startup: 'Bot state loaded'"
                    return $true
                }
            }
        }
        Start-Sleep -Seconds $PollInterval
    }
    Write-Error "[redeploy] FAILED: 'Bot state loaded' not seen in bot.log within $PollSeconds s"
    return $false
}

# 1) Pre-emptively kill any orphaned child recorded by the previous launcher.
foreach ($t in $Tasks) { Kill-ChildPid $t }

# 2) End the scheduled tasks (kills the launcher; child handled by watchdog).
foreach ($t in $Tasks) {
    Write-Host "[redeploy] ending task $t"
    & schtasks /End /TN $t 2>&1 | Out-Null
}
Start-Sleep -Seconds 3

# 3) (Re)kill orphans in case the new launcher hasn't acquired the pid file yet.
foreach ($t in $Tasks) { Kill-ChildPid $t }

# 4) Restart the tasks.
foreach ($t in $Tasks) {
    Write-Host "[redeploy] starting task $t"
    & schtasks /Run /TN $t 2>&1 | Out-Null
}

# 5) Verify the bot came back up cleanly.
$ok = Wait-ForBotLoaded
if (-not $ok) {
    foreach ($t in $Tasks) { Kill-ChildPid $t }
    exit 1
}
Write-Host "[redeploy] SUCCESS: $($Tasks -join ', ') redeployed and healthy."
