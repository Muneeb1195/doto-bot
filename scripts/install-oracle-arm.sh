#!/usr/bin/env bash
set -euo pipefail
# install-oracle-arm.sh — Full setup of Doto MT5 Bot on Oracle Cloud ARM64
#
# Prerequisites:
#   - Oracle Cloud Ampere A1 instance (Ubuntu 24.04, 4 OCPU, 24GB RAM)
#   - Git repo cloned to /home/ubuntu/doto-mt5-bot
#   - settings.ini and credentials.ini configured (broker login)
#
# Usage:
#   sudo bash scripts/install-oracle-arm.sh 2>&1 | tee install.log

REPO_DIR="/home/ubuntu/doto-mt5-bot"
cd "$REPO_DIR"

log() { echo -e "\e[1;32m[+] $*\e[0m"; }
warn() { echo -e "\e[1;33m[!] $*\e[0m"; }
err() { echo -e "\e[1;31m[-] $*\e[0m"; exit 1; }

# ──────────────────────────────────────────────
# Phase 0: System prerequisites
# ──────────────────────────────────────────────
log "Phase 0: System prerequisites"
apt-get update -qq
apt-get install -y -qq \
    xvfb xdotool \
    wget curl git \
    python3.12 python3.12-venv python3.12-dev \
    build-essential cmake \
    pkg-config \
    libgtk-3-dev libgdk-pixbuf2.0-dev libpango1.0-dev libcairo2-dev \
    2>&1 | tail -1

# ──────────────────────────────────────────────
# Phase 1: Hangover 11.4 (Wine for ARM64)
# ──────────────────────────────────────────────
log "Phase 1: Installing Hangover 11.4 (ARM64 Wine)"
HANGOVER_VER="11.4"
HANGOVER_URL="https://github.com/AndreRH/hangover/releases/download/v${HANGOVER_VER}"

cd /tmp
for pkg in \
    hangover-libwow64fex_${HANGOVER_VER}_arm64.deb \
    hangover-libarm64ecfex_${HANGOVER_VER}_arm64.deb \
    hangover-wowbox64_${HANGOVER_VER}_arm64.deb \
    hangover-wine_${HANGOVER_VER}~noble_arm64.deb; do
    if [ ! -f "$pkg" ]; then
        wget -q "$HANGOVER_URL/$pkg" || warn "Failed to download $pkg (may not exist yet)"
    fi
done

dpkg -i hangover-*.deb 2>/dev/null || true
apt-get install -f -y -qq 2>&1 | tail -1

# Verify Hangover installation
WINEPREFIX="$HOME/.wine" WINEARCH=win64 wine wineboot --init 2>/dev/null || true
KERNEL_FILE="$HOME/.wine/drive_c/windows/system32/kernel32.dll"
if [ -f "$KERNEL_FILE" ] && file "$KERNEL_FILE" | grep -q "Aarch64"; then
    log "Hangover verified: kernel32.dll is ARM64"
else
    warn "Hangover install may have issues — proceeding anyway"
fi

# ──────────────────────────────────────────────
# Phase 2: Install MT5 terminal under Wine
# ──────────────────────────────────────────────
log "Phase 2: Installing MetaTrader 5 terminal"
MT5_INSTALLER="/tmp/mt5setup.exe"
if [ ! -f "$MT5_INSTALLER" ]; then
    wget -q "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" \
        -O "$MT5_INSTALLER"
fi

Xvfb :99 -screen 0 1280x720x16 &>/dev/null &
XVFB_PID=$!
sleep 1

DISPLAY=:99 WINEPREFIX="$HOME/.wine" wine "$MT5_INSTALLER" /auto &
MT5_INSTALL_PID=$!
log "Waiting for MT5 install (up to 120s)..."
for i in $(seq 1 120); do
    if [ -d "$HOME/.wine/drive_c/Program Files/MetaTrader 5" ]; then
        log "MT5 installed successfully"
        break
    fi
    if ! kill -0 $MT5_INSTALL_PID 2>/dev/null; then
        # Installer finished — check again
        sleep 2
        if [ -d "$HOME/.wine/drive_c/Program Files/MetaTrader 5" ]; then
            log "MT5 installed successfully"
            break
        fi
        warn "MT5 installer exited early — continuing"
        break
    fi
    sleep 1
done

kill $XVFB_PID 2>/dev/null || true

# ──────────────────────────────────────────────
# Phase 3: Install Windows Python + MetaTrader5 under Wine
# ──────────────────────────────────────────────
log "Phase 3: Installing Windows Python + MetaTrader5 packages"
PYTHON_INSTALLER="/tmp/python-3.12.9-amd64.exe"
if [ ! -f "$PYTHON_INSTALLER" ]; then
    wget -q "https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe" \
        -O "$PYTHON_INSTALLER"
fi

DISPLAY=:99 Xvfb :99 -screen 0 1280x720x16 &>/dev/null &
XVFB_PID=$!
sleep 1

DISPLAY=:99 WINEPREFIX="$HOME/.wine" wine "$PYTHON_INSTALLER" \
    /quiet InstallAllUsers=1 PrependPath=1 Include_test=0 Include_tools=0 2>/dev/null
sleep 5

# Install MetaTrader5 + rpyc in Windows Python
WINEPREFIX="$HOME/.wine" wine python -m pip install --upgrade pip -q 2>/dev/null || true
WINEPREFIX="$HOME/.wine" wine python -m pip install MetaTrader5 rpyc -q 2>/dev/null || true

kill $XVFB_PID 2>/dev/null || true

# ──────────────────────────────────────────────
# Phase 4: Native ARM Python venv + bot deps
# ──────────────────────────────────────────────
log "Phase 4: Native Python venv + dependencies"
python3.12 -m venv "$REPO_DIR/.venv"
source "$REPO_DIR/.venv/bin/activate"

# Install mt5linux first (RPyC bridge client)
pip install --upgrade pip -q
pip install mt5linux -q

# Install bot dependencies
pip install -r "$REPO_DIR/requirements.txt" -q 2>/dev/null || \
pip install \
    pandas numpy scikit-learn xgboost lightgbm \
    fastapi uvicorn python-multipart \
    schedule \
    joblib \
    -q

log "Python dependencies installed"

# ──────────────────────────────────────────────
# Phase 5: Create systemd services
# ──────────────────────────────────────────────
log "Phase 5: Installing systemd services"
SYSTEMD_DIR="/etc/systemd/system"

# Xvfb virtual display
cat > "$SYSTEMD_DIR/xvfb-mt5.service" << 'SVC'
[Unit]
Description=Xvfb virtual framebuffer for MT5
Before=mt5.service mt5-rpyc.service

[Service]
Type=simple
ExecStart=/usr/bin/Xvfb :99 -screen 0 1280x720x16
Restart=always
RestartSec=5
User=ubuntu

[Install]
WantedBy=multi-user.target
SVC

# MT5 terminal (Wine + Hangover)
cat > "$SYSTEMD_DIR/mt5.service" << 'SVC'
[Unit]
Description=MetaTrader 5 Terminal (Wine)
After=xvfb-mt5.service
Requires=xvfb-mt5.service
BindsTo=mt5-rpyc.service

[Service]
Type=forking
User=ubuntu
Environment=DISPLAY=:99
Environment=WINEPREFIX=/home/ubuntu/.wine
Environment=WINEARCH=win64
ExecStartPre=/bin/sleep 5
ExecStart=/home/ubuntu/doto-mt5-bot/scripts/start-mt5.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SVC

# RPyC bridge (Windows Python running mt5linux server)
cat > "$SYSTEMD_DIR/mt5-rpyc.service" << 'SVC'
[Unit]
Description=MT5 RPyC Bridge (mt5linux)
After=mt5.service
Requires=mt5.service

[Service]
Type=simple
User=ubuntu
Environment=DISPLAY=:99
Environment=WINEPREFIX=/home/ubuntu/.wine
Environment=WINEARCH=win64
ExecStart=/home/ubuntu/doto-mt5-bot/scripts/start-rpyc.sh
Restart=always
RestartSec=15
StartLimitInterval=300
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
SVC

# Bot main loop
cat > "$SYSTEMD_DIR/doto-bot.service" << 'SVC'
[Unit]
Description=Doto MT5 Trading Bot
After=mt5-rpyc.service
Requires=mt5-rpyc.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/doto-mt5-bot
ExecStart=/home/ubuntu/doto-mt5-bot/.venv/bin/python bot/main.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/ubuntu/doto-mt5-bot/logs/bot.log
StandardError=append:/home/ubuntu/doto-mt5-bot/logs/bot.log

[Install]
WantedBy=multi-user.target
SVC

# Dashboard (FastAPI)
cat > "$SYSTEMD_DIR/doto-dashboard.service" << 'SVC'
[Unit]
Description=Doto MT5 Dashboard
After=mt5-rpyc.service
Requires=mt5-rpyc.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/doto-mt5-bot
ExecStart=/home/ubuntu/doto-mt5-bot/.venv/bin/python -m uvicorn dashboard.api:app --host 0.0.0.0 --port 8501
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/ubuntu/doto-mt5-bot/logs/dashboard.log
StandardError=append:/home/ubuntu/doto-mt5-bot/logs/dashboard.log

[Install]
WantedBy=multi-user.target
SVC

# News sentiment service
cat > "$SYSTEMD_DIR/doto-news.service" << 'SVC'
[Unit]
Description=Doto News Sentiment Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/doto-mt5-bot
ExecStart=/home/ubuntu/doto-mt5-bot/.venv/bin/python services/news_sentiment.py
Restart=on-failure
RestartSec=30
StandardOutput=append:/home/ubuntu/doto-mt5-bot/logs/news.log
StandardError=append:/home/ubuntu/doto-mt5-bot/logs/news.log

[Install]
WantedBy=multi-user.target
SVC

log "Systemd services created"

# ──────────────────────────────────────────────
# Phase 6: Create timer units (cron equivalents)
# ──────────────────────────────────────────────
log "Phase 6: Installing systemd timers"

# Auto-optimizer — weekly Sunday 02:00
cat > "$SYSTEMD_DIR/doto-optimizer.service" << 'SVC'
[Unit]
Description=Doto Auto-Optimizer (weekly)

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/doto-mt5-bot
ExecStart=/home/ubuntu/doto-mt5-bot/.venv/bin/python bot/auto_optimizer.py --apply
StandardOutput=append:/home/ubuntu/doto-mt5-bot/logs/auto_optimizer.log
StandardError=append:/home/ubuntu/doto-mt5-bot/logs/auto_optimizer.log

[Install]
WantedBy=multi-user.target
SVC

cat > "$SYSTEMD_DIR/doto-optimizer.timer" << 'SVC'
[Unit]
Description=Run auto-optimizer weekly on Sunday 02:00

[Timer]
OnCalendar=Sun 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
SVC

# ML retrain — weekly Sunday 03:00
cat > "$SYSTEMD_DIR/doto-retrain.service" << 'SVC'
[Unit]
Description=Doto ML Model Retraining

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/doto-mt5-bot
ExecStartPre=/bin/sleep 3600
ExecStart=/home/ubuntu/doto-mt5-bot/.venv/bin/python bot/train_model.py --retrain-all --symbols ALL --years 3
StandardOutput=append:/home/ubuntu/doto-mt5-bot/logs/retrain.log
StandardError=append:/home/ubuntu/doto-mt5-bot/logs/retrain.log

[Install]
WantedBy=multi-user.target
SVC

cat > "$SYSTEMD_DIR/doto-retrain.timer" << 'SVC'
[Unit]
Description=Run ML retraining weekly on Sunday 03:00

[Timer]
OnCalendar=Sun 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
SVC

# Backup — daily 04:00
cat > "$SYSTEMD_DIR/doto-backup.service" << 'SVC'
[Unit]
Description=Doto Daily Backup

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/doto-mt5-bot
ExecStart=/home/ubuntu/doto-mt5-bot/.venv/bin/python bot/backup.py
StandardOutput=append:/home/ubuntu/doto-mt5-bot/logs/backup.log
StandardError=append:/home/ubuntu/doto-mt5-bot/logs/backup.log

[Install]
WantedBy=multi-user.target
SVC

cat > "$SYSTEMD_DIR/doto-backup.timer" << 'SVC'
[Unit]
Description=Run daily backup at 04:00

[Timer]
OnCalendar=04:00:00
Persistent=true

[Install]
WantedBy=timers.target
SVC

# Weekly summary — Monday 05:00
cat > "$SYSTEMD_DIR/doto-weekly-summary.service" << 'SVC'
[Unit]
Description=Doto Weekly Summary

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/doto-mt5-bot
ExecStart=/home/ubuntu/doto-mt5-bot/.venv/bin/python scripts/weekly_summary.py
StandardOutput=append:/home/ubuntu/doto-mt5-bot/logs/weekly.log
StandardError=append:/home/ubuntu/doto-mt5-bot/logs/weekly.log

[Install]
WantedBy=multi-user.target
SVC

cat > "$SYSTEMD_DIR/doto-weekly-summary.timer" << 'SVC'
[Unit]
Description=Run weekly summary on Monday 05:00

[Timer]
OnCalendar=Mon 05:00:00
Persistent=true

[Install]
WantedBy=timers.target
SVC

log "Systemd timers created"

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
    sudo systemctl restart "$svc" 2>/dev/null || sudo systemctl start "$svc"
done

# Wait and check bot health
sleep 15
if systemctl is-active --quiet doto-bot; then
    if grep -q "Bot state loaded" /home/ubuntu/doto-mt5-bot/logs/bot.log 2>/dev/null; then
        log "SUCCESS: doto-bot healthy"
    else
        log "WARNING: doto-bot running but 'Bot state loaded' not found in log"
    fi
else
    log "FAILED: doto-bot not running"
    sudo journalctl -u doto-bot --no-pager -n 20
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
cd /home/ubuntu/doto-mt5-bot
git pull
exec bash scripts/redeploy.sh
SCRIPT
chmod +x "$REPO_DIR/scripts/update-and-redeploy.sh"

log "Wrapper scripts created"

# ──────────────────────────────────────────────
# Phase 8: Configure settings.ini for Wine paths
# ──────────────────────────────────────────────
log "Phase 8: Configuring settings.ini for Linux/Wine"
# Update MT5 path in config (if using settings.ini template)
if [ -f "$REPO_DIR/config/settings.ini" ]; then
    sed -i 's|path = .*|path = /home/ubuntu/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe|' \
        "$REPO_DIR/config/settings.ini" 2>/dev/null || true
fi

# ──────────────────────────────────────────────
# Phase 9: Enable and start services
# ──────────────────────────────────────────────
log "Phase 9: Enabling and starting services"
systemctl daemon-reload

systemctl enable xvfb-mt5.service
systemctl enable mt5.service
systemctl enable mt5-rpyc.service
systemctl enable doto-bot.service
systemctl enable doto-dashboard.service
systemctl enable doto-news.service

systemctl enable doto-optimizer.timer
systemctl enable doto-retrain.timer
systemctl enable doto-backup.timer
systemctl enable doto-weekly-summary.timer

log "Starting services (this will take ~60s)..."
systemctl start xvfb-mt5.service
sleep 3
systemctl start mt5.service
sleep 20  # MT5 takes time to start under Wine
systemctl start mt5-rpyc.service
sleep 10
systemctl start doto-bot.service
systemctl start doto-dashboard.service
systemctl start doto-news.service

systemctl start doto-optimizer.timer
systemctl start doto-retrain.timer
systemctl start doto-backup.timer
systemctl start doto-weekly-summary.timer

# ──────────────────────────────────────────────
# Phase 10: Verify deployment
# ──────────────────────────────────────────────
log "Phase 10: Verifying deployment"
echo ""
echo "=== Service Status ==="
for svc in xvfb-mt5 mt5 mt5-rpyc doto-bot doto-dashboard doto-news; do
    status=$(systemctl is-active "$svc" 2>/dev/null || echo "inactive")
    echo "  $svc: $status"
done

echo ""
echo "=== Timer Status ==="
for timer in doto-optimizer doto-retrain doto-backup doto-weekly-summary; do
    status=$(systemctl is-active "$timer.timer" 2>/dev/null || echo "inactive")
    next=$(systemctl show "$timer.timer" -p NextElapseUSecRealtime --value 2>/dev/null || echo "unknown")
    echo "  $timer: $status (next: $next)"
done

echo ""
log "Installation complete!"
echo ""
echo "Useful commands:"
echo "  sudo journalctl -u doto-bot -n 50 --no-pager    # Bot logs"
echo "  sudo journalctl -u mt5 -n 20 --no-pager         # MT5 terminal logs"
echo "  sudo systemctl status doto-bot                   # Bot status"
echo "  bash scripts/redeploy.sh                          # Redeploy after git pull"
echo ""
echo "NOTE: MT5 may need broker login on first run."
echo "Access the dashboard at http://<vps-ip>:8501 once running."
