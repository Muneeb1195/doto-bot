# Doto MT5 Bot - Windows Task Scheduler Setup
# Run as Administrator: powershell -ExecutionPolicy Bypass -File import_tasks.ps1

$RepoDir = "C:\doto-mt5-bot"
$Python = Join-Path $RepoDir ".venv\Scripts\python.exe"
$BotScript = Join-Path $RepoDir "migration\run_bot.ps1"
$DashScript = Join-Path $RepoDir "migration\run_dashboard.ps1"
$NewsScript = Join-Path $RepoDir "migration\run_news.ps1"
$OptimizerScript = Join-Path $RepoDir "bot\optimize_params.py"
$AutoOptimizerScript = Join-Path $RepoDir "bot\auto_optimizer.py"
$RetrainScript = Join-Path $RepoDir "bot\train_model.py"
$BackupScript = Join-Path $RepoDir "bot\backup.py"
$SummaryScript = Join-Path $RepoDir "bot\weekly_summary.py"

$CurrentUser = "$env:USERDOMAIN\$env:USERNAME"

function New-DotoTask {
    param($Name, $Execute, $Argument, $TriggerType, $RestartOnFailure, $At, $Hidden = $false)
    $action = New-ScheduledTaskAction -Execute $Execute -Argument $Argument

    if ($TriggerType -eq "boot") {
        $trigger = New-ScheduledTaskTrigger -AtStartup
    } elseif ($TriggerType -eq "logon") {
        $trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser
    } elseif ($TriggerType -eq "daily") {
        $trigger = New-ScheduledTaskTrigger -Daily -At $At
    } elseif ($TriggerType -eq "weekly") {
        $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $At.Split(" ")[0] -At $At.Split(" ")[1]
    }

    if ($RestartOnFailure) {
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -Hidden:$Hidden -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    } else {
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -Hidden:$Hidden
    }

    $principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "  [+] $Name"
}

Write-Host "=== Doto Bot - Task Scheduler Setup ==="

$PsArgs = "-ExecutionPolicy Bypass -File `"$BotScript`""
New-DotoTask -Name "DotoBot" -Execute "powershell.exe" -Argument $PsArgs -TriggerType "logon" -RestartOnFailure $true -Hidden $true
$PsArgs = "-ExecutionPolicy Bypass -File `"$DashScript`""
New-DotoTask -Name "DotoDashboard" -Execute "powershell.exe" -Argument $PsArgs -TriggerType "logon" -RestartOnFailure $true -Hidden $true
$PsArgs = "-ExecutionPolicy Bypass -File `"$NewsScript`""
New-DotoTask -Name "DotoNewsSentiment" -Execute "powershell.exe" -Argument $PsArgs -TriggerType "logon" -RestartOnFailure $true -Hidden $true

$OptArgs = "`"$AutoOptimizerScript`" --apply"
New-DotoTask -Name "DotoOptimizer" -Execute $Python -Argument $OptArgs -TriggerType "daily" -At "02:00"
$BackupArgs = "`"$BackupScript`""
New-DotoTask -Name "DotoBackup" -Execute $Python -Argument $BackupArgs -TriggerType "daily" -At "04:00"

$RetrainArgs = "`"$RetrainScript`" --retrain-all"
New-DotoTask -Name "DotoRetrain" -Execute $Python -Argument $RetrainArgs -TriggerType "weekly" -At "Sunday 03:00"
$SummaryArgs = "`"$SummaryScript`""
New-DotoTask -Name "DotoWeeklySummary" -Execute $Python -Argument $SummaryArgs -TriggerType "weekly" -At "Monday 05:00"

Write-Host "=== All tasks registered. ==="
Write-Host "Verify: Get-ScheduledTask -TaskName Doto*"
Write-Host "Add start_mt5.cmd to shell:startup so MT5 launches on login."
