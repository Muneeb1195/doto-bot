# Doto MT5 Bot — Windows Migration Guide

## Prerequisites
- Windows Desktop (10 or 11)
- Internet connection
- ~30 min of hands-on time

## Step 1: Install Required Software

| Software | Link / Command | Notes |
|----------|---------------|-------|
| Python 3.12 | [python.org](https://www.python.org/downloads/) | Check "Add Python to PATH", choose "Install for all users" |
| Git | [git-scm.com](https://git-scm.com/download/win) | Default options fine |
| MetaTrader 5 | Download from your broker | Log in with your credentials |

## Step 2: Clone Repository

```cmd
cd C:\
git clone <your-repo-url> doto-mt5-bot
cd doto-mt5-bot
```

## Step 3: Create Virtual Environments & Install Dependencies

```cmd
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

python -m venv .venv_news
.venv_news\Scripts\pip install -r requirements.txt
```

## Step 4: Copy Data Files

Copy these files/directories from your Linux installation (`~/doto-mt5-bot/`) to `C:\doto-mt5-bot\`:

| From (Linux) | To (Windows) |
|-------------|--------------|
| `config/settings.ini` | `C:\doto-mt5-bot\config\settings.ini` |
| `config/credentials.ini` | `C:\doto-mt5-bot\config\credentials.ini` |
| `models/*.pkl` | `C:\doto-mt5-bot\models\*.pkl` |
| `models/*.calib.npz` | `C:\doto-mt5-bot\models\*.calib.npz` |
| `data/bot_state.json` | `C:\doto-mt5-bot\data\bot_state.json` |
| `data/dashboard_state.json` | `C:\doto-mt5-bot\data\dashboard_state.json` |
| `logs/trades.csv` | `C:\doto-mt5-bot\logs\trades.csv` |

## Step 5: Verify

```cmd
cd C:\doto-mt5-bot
.venv\Scripts\python -m pytest tests/ -v
```

All ~249 tests should pass. If any MT5-related tests fail, make sure MetaTrader 5 is running and logged in.

## Step 6: Register Task Scheduler Tasks

1. Open **PowerShell as Administrator**
2. Run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
C:\doto-mt5-bot\migration\import_tasks.ps1
```

This registers 7 tasks:
- **DotoBot** — main trading loop (auto-restart on crash)
- **DotoDashboard** — dashboard API (auto-restart on crash)
- **DotoNewsSentiment** — news sentiment service (auto-restart on crash)
- **DotoOptimizer** — daily 02:00 parameter optimization
- **DotoRetrain** — weekly Sunday 03:00 ML retraining
- **DotoBackup** — daily 04:00 backup
- **DotoWeeklySummary** — weekly Monday 05:00 summary

## Step 7: Auto-start MT5 Terminal

1. Press `Win + R`, type `shell:startup`, press Enter
2. Copy `C:\doto-mt5-bot\migration\start_mt5.cmd` into the folder
3. This starts MT5 minimized to tray on every login

## Step 8: Reboot

Reboot your PC. Windows auto-logs in (if configured), MT5 starts, all bot services start within 60 seconds.

## Verify it's running

After login, check:
```powershell
Get-ScheduledTask -TaskName "Doto*" | Format-Table TaskName,State
```

Open `C:\doto-mt5-bot\logs\bot.log` — should show signals cycling.
Open `http://localhost:8501` in browser — dashboard should be live.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Python not found" | Reinstall Python, check "Add to PATH" |
| MT5 fails to connect | Make sure terminal64.exe is running and logged in |
| Bot crashes immediately | Check `logs/bot.log` for error trace |
| Task Scheduler won't import | Run PowerShell as Administrator |
| Permission denied on .venv | `icacls C:\doto-mt5-bot /grant Users:(OI)(CI)F /T` |
| Wrong Python version | `python --version` must be 3.12.x |
