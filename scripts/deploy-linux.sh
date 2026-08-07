#!/usr/bin/env bash
# deploy-linux.sh — Full setup of Doto MT5 Bot on x86_64 Linux (CachyOS / Arch)
#
# Adapted from install-oracle-arm.sh for Arch-based distros:
#   - pacman instead of apt
#   - native Wine (x86_64) — NO Hangover/ARM needed
#   - systemd USER services (no root; uses loginctl enable-linger)
#
# Prerequisites:
#   - CachyOS/Arch x86_64 with Tailscale + SSH (headless is fine)
#   - Git repo copied to ~/doto-mt5-bot (rsync or git clone)
#   - config/credentials.ini placed manually (broker login) — NEVER committed
#   - config/settings.ini reviewed (portfolio, broker, model paths)
#
# Usage:
#   bash scripts/deploy-linux.sh 2>&1 | tee deploy.log
#
# Notes:
#   - Requires the [multilib] repo enabled in /etc/pacman.conf for Wine.
#   - MT5 terminal runs under Wine on the Xvfb :99 virtual display.
#   - The native venv talks to MT5 through the mt5linux RPyC bridge on
#     127.0.0.1:18812 (Wine-hosted Windows Python runs the bridge server).

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/doto-mt5-bot}"
RUN_USER="${RUN_USER:-$(id -un)}"
HOME_DIR="$(eval echo "~$RUN_USER")"
WINEPREFIX="$HOME_DIR/.wine"
SYSTEMD_USER_DIR="$HOME_DIR/.config/systemd/user"

log() { echo -e "\e[1;32m[+] $*\e[0m"; }
warn() { echo -e "\e[1;33m[!] $*\e[0m"; }
err() { echo -e "\e[1;31m[-] $*\e[0m"; exit 1; }

command -v sudo >/dev/null 2>&1 || err "sudo is required"

# ──────────────────────────────────────────────
# Phase 0: System prerequisites (pacman)
# ──────────────────────────────────────────────
log "Phase 0: System prerequisites"
if ! command -v wine >/dev/null 2>&1; then
    log "Enabling [multilib] repo for Wine"
    sudo sed -i 's|^#\[multilib\]|\[multilib\]|; s|^#Include = /etc/pacman.d/mirrorlist$|Include = /etc/pacman.d/mirrorlist|' \
        /etc/pacman.conf 2>/dev/null || warn "Could not auto-enable multilib — enable it manually"
fi

sudo pacman -S --noconfirm --needed \
    wine \
    xorg-xvfb xdotool \
    wget curl git \
    python python-virtualenv python-pip \
    base-devel cmake pkg-config \
    2>&1 || warn "pacman reported issues — continuing"

# ──────────────────────────────────────────────
# Phase 1: Initialize Wine prefix
# ──────────────────────────────────────────────
log "Phase 1: Initializing Wine prefix (win64)"
export WINEPREFIX="$WINEPREFIX"
export WINEARCH=win64
wineboot --init 2>/dev/null || warn "wineboot reported issues — continuing"
log "Wine prefix ready at $WINEPREFIX"

# ──────────────────────────────────────────────
# Phase 2: Install MT5 terminal under Wine
# ──────────────────────────────────────────────
log "Phase 2: Installing MetaTrader 5 terminal"
MT5_INSTALLER="/tmp/mt5setup.exe"
MT5_DIR="$WINEPREFIX/drive_c/Program Files/MetaTrader 5"
if [ ! -f "$MT5_INSTALLER" ]; then
    wget -q "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" \
        -O "$MT5_INSTALLER"
fi

Xvfb :99 -screen 0 1280x720x16 &>/dev/null &
XVFB_PID=$!
sleep 1

DISPLAY=:99 wine "$MT5_INSTALLER" /auto &
MT5_INSTALL_PID=$!
log "Waiting for MT5 install (up to 120s)..."
for i in $(seq 1 120); do
    if [ -d "$MT5_DIR" ]; then
        log "MT5 installed successfully"
        break
    fi
    if ! kill -0 $MT5_INSTALL_PID 2>/dev/null; then
        sleep 2
        if [ -d "$MT5_DIR" ]; then
            log "MT5 installed successfully"
            break
        fi
        warn "MT5 installer exited early — continuing"
        break
    fi
    sleep 1
done

[ -d "$MT5_DIR" ] || err "MT5 install failed — MT5 dir missing at $MT5_DIR"
kill $XVFB_PID 2>/dev/null || true

# ──────────────────────────────────────────────
# Phase 3: Install Windows Python + MT5 bridge deps under Wine
# ──────────────────────────────────────────────
log "Phase 3: Installing Windows Python + MetaTrader5 packages"
PYTHON_INSTALLER="/tmp/python-3.12.9-amd64.exe"
if [ ! -f "$PYTHON_INSTALLER" ]; then
    wget -q "https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe" \
        -O "$PYTHON_INSTALLER"
fi

Xvfb :99 -screen 0 1280x720x16 &>/dev/null &
XVFB_PID=$!
sleep 1

DISPLAY=:99 wine "$PYTHON_INSTALLER" \
    /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 Include_tools=0 2>/dev/null
sleep 5

# Install MetaTrader5 + mt5linux (bridge server) + rpyc in Windows Python.
# mt5linux is REQUIRED — start-rpyc.sh runs `wine python -m mt5linux`.
wine python -m pip install --upgrade pip -q 2>/dev/null || true
wine python -m pip install MetaTrader5 mt5linux rpyc -q 2>/dev/null || true

kill $XVFB_PID 2>/dev/null || true

# ──────────────────────────────────────────────
# Phase 4: Native Linux Python venv + bot deps
# ──────────────────────────────────────────────
log "Phase 4: Native Python venv + dependencies"
python -m venv "$REPO_DIR/.venv"
source "$REPO_DIR/.venv/bin/activate"

# Install mt5linux FIRST (RPyC bridge client for the native side)
pip install --upgrade pip -q
pip install mt5linux -q

# Install bot dependencies
if [ -f "$REPO_DIR/requirements.txt" ]; then
    pip install -r "$REPO_DIR/requirements.txt" -q 2>/dev/null || \
        pip install pandas numpy scikit-learn xgboost lightgbm fastapi uvicorn python-multipart joblib -q
else
    pip install pandas numpy scikit-learn xgboost lightgbm fastapi uvicorn python-multipart joblib -q
fi

log "Python dependencies installed"

# ──────────────────────────────────────────────
# Phase 5: Create systemd USER services
# ──────────────────────────────────────────────
log "Phase 5: Installing systemd user services"
mkdir -p "$SYSTEMD_USER_DIR"

# Xvfb virtual display
cat > "$SYSTEMD_USER_DIR/xvfb-mt5.service" << SVC
[Unit]
Description=Xvfb virtual framebuffer for MT5
Before=mt5.service mt5-rpyc.service

[Service]
Type=simple
ExecStart=/usr/bin/Xvfb :99 -screen 0 1280x720x16
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
SVC

# MT5 terminal (Wine)
cat > "$SYSTEMD_USER_DIR/mt5.service" << SVC
[Unit]
Description=MetaTrader 5 Terminal (Wine)
After=xvfb-mt5.service
Requires=xvfb-mt5.service

[Service]
Type=forking
Environment=DISPLAY=:99
Environment=WINEPREFIX=$WINEPREFIX
Environment=WINEARCH=win64
ExecStartPre=/bin/sleep 5
ExecStart=$REPO_DIR/scripts/start-mt5.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
SVC

# RPyC bridge (Windows Python running mt5linux server)
cat > "$SYSTEMD_USER_DIR/mt5-rpyc.service" << SVC
[Unit]
Description=MT5 RPyC Bridge (mt5linux)
After=mt5.service
Requires=mt5.service

[Service]
Type=simple
Environment=DISPLAY=:99
Environment=WINEPREFIX=$WINEPREFIX
Environment=WINEARCH=win64
ExecStart=$REPO_DIR/scripts/start-rpyc.sh
Restart=always
RestartSec=15
StartLimitIntervalSec=300
StartLimitBurst=3

[Install]
WantedBy=default.target
SVC

# Bot main loop
cat > "$SYSTEMD_USER_DIR/doto-bot.service" << SVC
[Unit]
Description=Doto MT5 Trading Bot
After=mt5-rpyc.service
Requires=mt5-rpyc.service

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/.venv/bin/python bot/main.py
Restart=on-failure
RestartSec=10
StandardOutput=append:$REPO_DIR/logs/bot.log
StandardError=append:$REPO_DIR/logs/bot.log

[Install]
WantedBy=default.target
SVC

# Dashboard (FastAPI — private tier)
cat > "$SYSTEMD_USER_DIR/doto-dashboard.service" << SVC
[Unit]
Description=Doto MT5 Dashboard
After=mt5-rpyc.service
Requires=mt5-rpyc.service

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
Environment=DASHBOARD_USER=\$DASHBOARD_USER
Environment=DASHBOARD_PASS=\$DASHBOARD_PASS
ExecStart=$REPO_DIR/.venv/bin/python -m uvicorn dashboard.api:app --host 0.0.0.0 --port 8501
Restart=on-failure
RestartSec=10
StandardOutput=append:$REPO_DIR/logs/dashboard.log
StandardError=append:$REPO_DIR/logs/dashboard.log

[Install]
WantedBy=default.target
SVC

# News sentiment service
cat > "$SYSTEMD_USER_DIR/doto-news.service" << SVC
[Unit]
Description=Doto News Sentiment Service
After=network.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/.venv/bin/python services/news_sentiment.py
Restart=on-failure
RestartSec=30
StandardOutput=append:$REPO_DIR/logs/news.log
StandardError=append:$REPO_DIR/logs/news.log

[Install]
WantedBy=default.target
SVC

log "Systemd user services created"

# ──────────────────────────────────────────────
# Phase 6: Create timer units (cron equivalents)
# ──────────────────────────────────────────────
log "Phase 6: Installing systemd user timers"

# Auto-optimizer — weekly Sunday 02:00 (full-grid 2yr, applies + restarts bot)
cat > "$SYSTEMD_USER_DIR/doto-optimizer.service" << SVC
[Unit]
Description=Doto Auto-Optimizer (weekly)

[Service]
Type=oneshot
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/.venv/bin/python bot/auto_optimizer.py --mode weekly --apply
StandardOutput=append:$REPO_DIR/logs/auto_optimizer.log
StandardError=append:$REPO_DIR/logs/auto_optimizer.log

[Install]
WantedBy=default.target
SVC

cat > "$SYSTEMD_USER_DIR/doto-optimizer.timer" << SVC
[Unit]
Description=Run auto-optimizer weekly on Sunday 02:00

[Timer]
OnCalendar=Sun 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
SVC

# ML retrain — weekly Sunday 03:00
cat > "$SYSTEMD_USER_DIR/doto-retrain.service" << SVC
[Unit]
Description=Doto ML Model Retraining

[Service]
Type=oneshot
WorkingDirectory=$REPO_DIR
ExecStartPre=/bin/sleep 3600
ExecStart=$REPO_DIR/.venv/bin/python bot/train_model.py --retrain-all --symbols ALL --years 3
StandardOutput=append:$REPO_DIR/logs/retrain.log
StandardError=append:$REPO_DIR/logs/retrain.log

[Install]
WantedBy=default.target
SVC

cat > "$SYSTEMD_USER_DIR/doto-retrain.timer" << SVC
[Unit]
Description=Run ML retraining weekly on Sunday 03:00

[Timer]
OnCalendar=Sun 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
SVC

# Backup — daily 04:00
cat > "$SYSTEMD_USER_DIR/doto-backup.service" << SVC
[Unit]
Description=Doto Daily Backup

[Service]
Type=oneshot
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/.venv/bin/python bot/backup.py
StandardOutput=append:$REPO_DIR/logs/backup.log
StandardError=append:$REPO_DIR/logs/backup.log

[Install]
WantedBy=default.target
SVC

cat > "$SYSTEMD_USER_DIR/doto-backup.timer" << SVC
[Unit]
Description=Run daily backup at 04:00

[Timer]
OnCalendar=04:00:00
Persistent=true

[Install]
WantedBy=timers.target
SVC

cat > "$SYSTEMD_USER_DIR/doto-dashboard-publish.service" << SVC
[Unit]
Description=Doto Public Dashboard Publisher

[Service]
Type=oneshot
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/.venv/bin/python scripts/publish_dashboard.py
StandardOutput=append:$REPO_DIR/logs/publish_dashboard.log
StandardError=append:$REPO_DIR/logs/publish_dashboard.log

[Install]
WantedBy=default.target
SVC

cat > "$SYSTEMD_USER_DIR/doto-dashboard-publish.timer" << SVC
[Unit]
Description=Publish sanitized dashboard snapshot every 5 minutes

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
SVC

log "Systemd user timers created (weekly-summary skipped — no script)"

# ──────────────────────────────────────────────
# Phase 7: Create wrapper scripts
# ──────────────────────────────────────────────
log "Phase 7: Creating wrapper scripts"

# MT5 startup wrapper
cat > "$REPO_DIR/scripts/start-mt5.sh" << 'SCRIPT'
#!/usr/bin/env bash
# start-mt5.sh — Launch MT5 terminal under Wine (headless)
# Called by systemd (mt5.service)
export DISPLAY=:99
export WINEPREFIX="$HOME/.wine"
export WINEARCH=win64

MT5_PATH="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"

if [ ! -f "$MT5_PATH" ]; then
    echo "MT5 terminal not found at $MT5_PATH"
    exit 1
fi

# The /portable flag is essential — without it MT5 ignores ini configs
wine "$MT5_PATH" /portable
SCRIPT
chmod +x "$REPO_DIR/scripts/start-mt5.sh"

# RPyC bridge startup wrapper
cat > "$REPO_DIR/scripts/start-rpyc.sh" << 'SCRIPT'
#!/usr/bin/env bash
# start-rpyc.sh — Launch mt5linux RPyC server under Wine
# Called by systemd (mt5-rpyc.service)
export DISPLAY=:99
export WINEPREFIX="$HOME/.wine"
export WINEARCH=win64

# Wait for MT5 terminal to be ready
sleep 10

wine python -m mt5linux --port 18812 --host 127.0.0.1
SCRIPT
chmod +x "$REPO_DIR/scripts/start-rpyc.sh"

# Redeploy script
cat > "$REPO_DIR/scripts/redeploy.sh" << 'SCRIPT'
#!/usr/bin/env bash
# redeploy.sh — Restart bot + dashboard with health check
# Called after code changes (git pull)
set -euo pipefail

log() { echo "[redeploy] $*"; }

SERVICES=("doto-bot" "doto-dashboard")

for svc in "${SERVICES[@]}"; do
    log "restarting $svc"
    systemctl --user restart "$svc" 2>/dev/null || systemctl --user start "$svc"
done

# Wait and check bot health
sleep 15
if systemctl --user is-active --quiet doto-bot; then
    if grep -q "Bot state loaded" "$HOME/doto-mt5-bot/logs/bot.log" 2>/dev/null; then
        log "SUCCESS: doto-bot healthy"
    else
        log "WARNING: doto-bot running but 'Bot state loaded' not found in log"
    fi
else
    log "FAILED: doto-bot not running"
    journalctl --user -u doto-bot --no-pager -n 20
    exit 1
fi

log "SUCCESS: all services redeployed"
SCRIPT
chmod +x "$REPO_DIR/scripts/redeploy.sh"

# Configure git to auto-pull on redeploy
cat > "$REPO_DIR/scripts/update-and-redeploy.sh" << 'SCRIPT'
#!/usr/bin/env bash
# update-and-redeploy.sh — Git pull + redeploy
set -euo pipefail
cd "$HOME/doto-mt5-bot"
git pull
exec bash scripts/redeploy.sh
SCRIPT
chmod +x "$REPO_DIR/scripts/update-and-redeploy.sh"

log "Wrapper scripts created"

# ──────────────────────────────────────────────
# Phase 8: Configure settings.ini for Wine paths
# ──────────────────────────────────────────────
log "Phase 8: Configuring settings.ini for Linux/Wine"
if [ -f "$REPO_DIR/config/settings.ini" ]; then
    sed -i "s|path = .*|path = $WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe|" \
        "$REPO_DIR/config/settings.ini" 2>/dev/null || warn "Could not rewrite settings.ini path — fix manually"
fi

# ──────────────────────────────────────────────
# Phase 9: Enable lingering + start services
# ──────────────────────────────────────────────
log "Phase 9: Enabling lingering + starting services"

# Linger keeps user services running without an active login session
sudo loginctl enable-linger "$RUN_USER" 2>/dev/null || warn "enable-linger failed — services may stop at logout"

systemctl --user daemon-reload

systemctl --user enable xvfb-mt5.service
systemctl --user enable mt5.service
systemctl --user enable mt5-rpyc.service
systemctl --user enable doto-bot.service
systemctl --user enable doto-dashboard.service
systemctl --user enable doto-news.service

systemctl --user enable doto-optimizer.timer
systemctl --user enable doto-retrain.timer
systemctl --user enable doto-backup.timer
systemctl --user enable doto-dashboard-publish.timer

log "Starting services (this will take ~60s)..."
systemctl --user start xvfb-mt5.service
sleep 3
systemctl --user start mt5.service
sleep 20  # MT5 takes time to start under Wine
systemctl --user start mt5-rpyc.service
sleep 10
systemctl --user start doto-bot.service
systemctl --user start doto-dashboard.service
systemctl --user start doto-news.service

systemctl --user start doto-optimizer.timer
systemctl --user start doto-retrain.timer
systemctl --user start doto-backup.timer
systemctl --user start doto-dashboard-publish.timer

# ──────────────────────────────────────────────
# Phase 10: Verify deployment
# ──────────────────────────────────────────────
log "Phase 10: Verifying deployment"
echo ""
echo "=== Service Status ==="
for svc in xvfb-mt5 mt5 mt5-rpyc doto-bot doto-dashboard doto-news; do
    status=$(systemctl --user is-active "$svc" 2>/dev/null || echo "inactive")
    echo "  $svc: $status"
done

echo ""
echo "=== Timer Status ==="
for timer in doto-optimizer doto-retrain doto-backup doto-dashboard-publish; do
    status=$(systemctl --user is-active "$timer.timer" 2>/dev/null || echo "inactive")
    next=$(systemctl --user show "$timer.timer" -p NextElapseUSecRealtime --value 2>/dev/null || echo "unknown")
    echo "  $timer: $status (next: $next)"
done

echo ""
log "Installation complete!"
echo ""
echo "Useful commands:"
echo "  journalctl --user -u doto-bot -n 50 --no-pager    # Bot logs"
echo "  journalctl --user -u mt5 -n 20 --no-pager         # MT5 terminal logs"
echo "  systemctl --user status doto-bot                  # Bot status"
echo "  bash scripts/redeploy.sh                          # Redeploy after git pull"
echo ""
echo "NOTE: MT5 may need broker login on first run."
echo "Access the dashboard at http://<tailscale-ip>:8501 once running."
echo "DASHBOARD_USER / DASHBOARD_PASS env vars must be set for dashboard auth."
echo "  systemctl --user set-environment DASHBOARD_USER=admin DASHBOARD_PASS=secret"
echo "  systemctl --user restart doto-dashboard"
