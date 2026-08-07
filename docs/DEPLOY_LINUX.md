# Deploying the Doto MT5 Bot to the home-server (homer) Server (CachyOS / Arch)

This runbook walks through deploying the bot to a headless x86_64 Arch-based
(CachyOS) box. The bot uses a **native Linux Python** process talking to an MT5
terminal running **under Wine** via the `mt5linux` RPyC bridge on
`127.0.0.1:18812`.

> **One-time manual steps are clearly marked.** Everything else is automated by
> `scripts/deploy-linux.sh`.

---

## 1. Prerequisites

- home-server reachable over Tailscale (SSH works).
- The repo is already transferred with `rsync` (see §2).
- `config/credentials.ini` exists on home-server **but is never committed to git**.
- A broker account whose MT5 login will be used on the Wine terminal.

Check the server:

```bash
home-server 'uname -m && cat /etc/os-release | head -3 && free -h'
```

Expected: `x86_64`, `CachyOS`/Arch, enough RAM for MT5 under Wine (~1 GB).

---

## 2. Transfer the repo (rsync, carries gitignored files)

The dev box has `models/*.pkl`, `config/credentials.ini`, and log/data files
that are gitignored but **required on home-server**. They travel by rsync, never by
git.

```bash
# From the dev box, inside the repo:
rsync -avz --exclude='.git' --exclude='.venv*' \
  --exclude='wine' --exclude='backups' --exclude='__pycache__' \
  -e ssh ./ home-server:~/doto-mt5-bot/
```

- Keeps `.venv*` off the wire (a fresh Linux venv is created on home-server).
- `backups/` can be large (hundreds of MB) — skip it; `bot/backup.py`
  regenerates archives on home-server.

If you later clone from the private git repo instead, models/credentials must
be rsynced separately:

```bash
rsync -avz -e ssh ./models home-server:~/doto-mt5-bot/models/
rsync -avz -e ssh ./config/credentials.ini home-server:~/doto-mt5-bot/config/
```

---

## 3. Place credentials (manual, one-time)

```bash
home-server
mkdir -p ~/doto-mt5-bot/config ~/doto-mt5-bot/logs ~/doto-mt5-bot/models
nano ~/doto-mt5-bot/config/credentials.ini
```

Format (matches `bot/start_mt5.py`):

```ini
[LOGIN]
account = <your MT5 account number>
password = <your MT5 password>
server = <broker server name>
```

Also set the two dashboard auth vars used by `dashboard/api.py`:

```bash
systemctl --user set-environment DASHBOARD_USER=<user>
systemctl --user set-environment DASHBOARD_PASS=<password>
systemctl --user restart doto-dashboard
```

> These are read by the dashboard service unit via `$DASHBOARD_USER` /
> `$DASHBOARD_PASS`. Without them the dashboard refuses to start
> (RuntimeError in `dashboard/api.py`).

---

## 4. Run the deploy script

```bash
home-server
cd ~/doto-mt5-bot
bash scripts/deploy-linux.sh
```

What it does (full detail in the script; `scripts/install-oracle-arm.sh` was the
original reference):

| Phase | Action |
|-------|--------|
| 0 | Enable `[multilib]`; `pacman -Syu` wine, xorg-xvfb, xdotool, wget, curl, git, python, python-virtualenv, base-devel, cmake, pkg-config |
| 1 | `wineboot --init` (WINEPREFIX=~/.wine, win64) |
| 2 | Download `mt5setup.exe`, silent-install under Xvfb `:99` → `~/.wine/drive_c/Program Files/MetaTrader 5/` |
| 3 | Install Windows Python 3.12 under Wine; `wine python -m pip install MetaTrader5 mt5linux rpyc` |
| 4 | Create native venv `~/.venv`; `pip install mt5linux` then `requirements.txt` |
| 5 | systemd **user** services: xvfb-mt5, mt5, mt5-rpyc, doto-bot, doto-dashboard, doto-news |
| 6 | systemd **user** timers: doto-optimizer (Sun 02:00, `--mode weekly --apply`), doto-retrain (Sun 03:00, `--years 3`), doto-backup (daily 04:00), doto-dashboard-publish (every 5 min) |
| 7 | Wrapper scripts `start-mt5.sh` (with `/portable`), `start-rpyc.sh` (`mt5linux` on :18812), `redeploy.sh`, `update-and-redeploy.sh` |
| 8 | Point `settings.ini` `[MT5] path` at the Wine terminal64.exe |
| 9 | `loginctl enable-linger`, daemon-reload, enable + start everything in order |
| 10 | Verify all services/timers are active |

---

## 5. First login to the MT5 terminal (manual, one-time)

The MT5 terminal under Wine needs to accept the broker login at least once.
If the broker requires additional authentication (e.g., OTP), do it by hand:

```bash
systemctl --user stop mt5-rpyc doto-bot   # pause consumers
Xvfb :99 -screen 0 1280x720x16 -nolisten tcp &
export DISPLAY=:99
wine "$HOME/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe" /portable
```

Log in inside the Wine GUI (or use `bot/start_mt5.py`), confirm balance, then:

```bash
kill %1; systemctl --user start doto-bot mt5-rpyc
```

---

## 6. Health checks

```bash
systemctl --user is-active xvfb-mt5 mt5 mt5-rpyc doto-bot doto-dashboard doto-news
journalctl --user -u doto-bot -n 50 --no-pager        # bot log
tail -f ~/doto-mt5-bot/logs/bot.log                   # "Bot state loaded" marker
curl -s http://localhost:8501/api/state | head -c 200  # dashboard API
```

Expected startup marker in `logs/bot.log`:

```
Bot state loaded
MT5 reconnected: <name> | Balance: Rs.<balance>
```

---

## 7. Daily operations

- **Update code**: `~/doto-mt5-bot/scripts/update-and-redeploy.sh` (git pull +
  restart bot + dashboard, waits for "Bot state loaded").
- **Restart bot**: `systemctl --user restart doto-bot`.
- **Check timers**: `systemctl --user list-timers`.
- **Optimization run**: weekly Sun 02:00 via doto-optimizer; the bot is paused
  (`stop_bot()`) before optimization and restarted after settings write.
- **Public dashboard**: `doto-dashboard-publish.timer` runs
  `scripts/publish_dashboard.py` every 5 minutes, pushing a sanitized snapshot
  to the public GitHub Pages site
  https://muneeb1195.github.io/doto-dashboard/ (balances, positions, and
  per-trade P&L are never published). Log: `logs/publish_dashboard.log`.
  One-time setup on home-server: `gh auth login` then
  `git clone -b gh-pages https://github.com/Muneeb1195/doto-dashboard.git .dashboard_public`
  inside the repo (the publisher clones it automatically on first run if missing).

---

## 8. Power outage recovery

Timers are `Persistent=true` and services use `Restart=on-failure`/`always`, so
everything comes back automatically after a reboot. If the box loses power:

```bash
# after reboot, verify everything is back:
systemctl --user list-units --type=service --state=active | grep doto
```

For prolonged outages, consider a UPS — the box is a low-power Intel NUC
(i5-4250U), a small UPS is sufficient to ride out brief dips.

---

## 9. Uninstall / cleanup

```bash
systemctl --user disable --now doto-optimizer.timer doto-retrain.timer doto-backup.timer \
  doto-dashboard-publish.timer \
  doto-bot doto-dashboard doto-news mt5-rpyc mt5 xvfb-mt5
rm -rf ~/.config/systemd/user/doto-* ~/.config/systemd/user/xvfb-mt5.service ~/.config/systemd/user/mt5.service
sudo loginctl disable-linger $USER
```

---

## Notes / known caveats

- `mt5linux` is **not** in `requirements.txt`; it is installed separately in
  both the native venv (client) and the Wine Python (server).
- The old Wine-Python launcher scripts (`bot/run_bot.sh`, `bot/run_optimizer.sh`,
  `bot/train_all.sh`) are **obsolete** — current `mt5_connect.py` always uses
  the mt5linux RPyC path on Linux.
- MT5 first connect under Wine can take 100+ seconds; the RPyC service
  (`Restart=always`, RestartSec=15) will keep retrying until MT5 is ready.
- The dashboard is two-tier: full data stays on home-server (`:8501`, reachable via
  `tailscale serve 8501`); a sanitized snapshot is published to GitHub Pages by
  `scripts/publish_dashboard.py` (§7 of this doc / see that script).
