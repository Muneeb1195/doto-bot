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
#   - The native venv talks to MT5 through the mt5_socket_server MQL5 EA,
#     which runs inside the terminal and listens on 127.0.0.1:9000.
#     (Wine's named pipes are broken, so the MetaTrader5 Python package and
#     every RPyC bridge built on it cannot work here — hence the EA.)

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
    xorg-server-xvfb xdotool \
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
# Generic MetaQuotes build — used for the base install. The broker-branded
# installer is fetched later (Phase 2b) purely for its servers.dat, which is
# the only place the DOTOGlobal-* server entries exist. Without it the
# terminal cannot log in and TERMINAL_CONNECTED stays false.
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

# ──────────────────────────────────────────────
# Phase 2b: Broker-branded servers.dat
# ──────────────────────────────────────────────
# The generic MetaQuotes build ships a servers.dat with no DOTOGlobal entries,
# so the terminal silently never attempts a login (zero "Network" log lines).
# Installing the branded build purely to harvest its servers.dat fixes this
# without disturbing the working MetaTrader 5 dir (EA, common.ini, .ex5).
log "Phase 2b: Installing broker-branded servers.dat"
BRAND_INSTALLER="/tmp/dotoglobal5setup.exe"
BRAND_DIR="$WINEPREFIX/drive_c/Program Files/DOTO Global MT5 Terminal"
if [ ! -f "$BRAND_INSTALLER" ]; then
    wget -q "https://download.mql5.com/cdn/web/21973/mt5/dotoglobal5setup.exe" \
        -O "$BRAND_INSTALLER" || warn "Branded installer download failed"
fi
if [ -f "$BRAND_INSTALLER" ] && [ ! -f "$BRAND_DIR/Config/servers.dat" ]; then
    DISPLAY=:99 timeout 300 wine "$BRAND_INSTALLER" /auto || \
        warn "Branded installer reported issues — continuing"
fi
if [ -f "$BRAND_DIR/Config/servers.dat" ]; then
    [ -f "$MT5_DIR/Config/servers.dat" ] && \
        cp "$MT5_DIR/Config/servers.dat" "$MT5_DIR/Config/servers.dat.generic.bak"
    cp "$BRAND_DIR/Config/servers.dat" "$MT5_DIR/Config/servers.dat"
    log "Broker servers.dat installed"
else
    warn "Broker servers.dat not found — MT5 will not be able to log in"
fi

kill $XVFB_PID 2>/dev/null || true

# ──────────────────────────────────────────────
# Phase 3: Install mt5server.exe (standalone RPyC bridge)
# ──────────────────────────────────────────────
log "Phase 3: Installing mt5server.exe (standalone MT5 RPyC bridge)"
MT5SERVER_URL="https://github.com/lucas-campagna/mt5linux/releases/download/server-1.1.1/mt5server.exe"
MT5SERVER_PATH="$REPO_DIR/scripts/mt5server.exe"

if [ ! -f "$MT5SERVER_PATH" ]; then
    log "Downloading mt5server.exe from GitHub releases..."
    wget -q "$MT5SERVER_URL" -O "$MT5SERVER_PATH" || \
        err "Failed to download mt5server.exe — check internet connection"
    chmod +x "$MT5SERVER_PATH"
    log "mt5server.exe downloaded ($(du -h "$MT5SERVER_PATH" | cut -f1))"
else
    log "mt5server.exe already present — skipping download"
fi

# NOTE: No Windows Python needed — mt5server.exe is a standalone binary
# that handles MT5 IPC internally (avoids Wine named-pipe bugs).

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
Before=mt5.service

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
# Type=simple + exec in start-mt5.sh: the terminal IS the main process, so
# systemd tracks a real MainPID. Type=forking left MainPID=0 and the unit
# reported inactive while the terminal was actually running, which defeated
# Restart=on-failure. The old ExecStartPre=/bin/sleep 5 was also not enough
# for Xvfb after a cold boot; the script now polls the display instead.
Type=simple
TimeoutStartSec=120
Environment=DISPLAY=:99
Environment=WINEPREFIX=$WINEPREFIX
Environment=WINEARCH=win64
ExecStart=$REPO_DIR/scripts/start-mt5.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
SVC

# Bot main loop
cat > "$SYSTEMD_USER_DIR/doto-bot.service" << SVC
[Unit]
Description=Doto MT5 Trading Bot
After=mt5.service
Requires=mt5.service

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
# Read dashboard credentials from environment (must be set before running this script)
DASHBOARD_USER="${DASHBOARD_USER:-admin}"
DASHBOARD_PASS="${DASHBOARD_PASS:-}"
if [ -z "$DASHBOARD_PASS" ] || [ "$DASHBOARD_PASS" = "changeme" ]; then
    echo "ERROR: DASHBOARD_PASS must be set to a non-default value before deploying."
    echo "  export DASHBOARD_PASS=your-secure-password"
    exit 1
fi
cat > "$SYSTEMD_USER_DIR/doto-dashboard.service" << SVC
[Unit]
Description=Doto MT5 Dashboard
After=mt5.service

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
Environment=DASHBOARD_USER=$DASHBOARD_USER
Environment=DASHBOARD_PASS=$DASHBOARD_PASS
ExecStart=$REPO_DIR/.venv/bin/python -m uvicorn dashboard.api:app --host 127.0.0.1 --port 8501
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

# NOTE: optimization and ML retraining deliberately have NO systemd timers.
# Both run on GitHub Actions only (.github/workflows/optimize.yml and
# train.yml). A single 48-combo symbol takes ~23 min on this box and the full
# portfolio ~8h, which would also compete with the live bot for CPU. CI shards
# the symbols across parallel jobs instead, and doto-download.timer pulls the
# resulting models/params back here.

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

# A leftover wineserver from the previous terminal (crashed, killed, or still
# shutting down during a `systemctl restart`) leaves the prefix in a state
# where the new terminal starts, never binds the EA socket, and exits after a
# few seconds -- systemd then restart-loops it. Always reap unconditionally and
# WAIT for the prefix to drain: a conditional reap raced the outgoing terminal
# on restart and reintroduced the failure.
echo "Draining wine prefix"
wineserver -k 2>/dev/null || true
for _ in $(seq 1 15); do
    pgrep -x wineserver >/dev/null 2>&1 || break
    sleep 1
done

# Wine must not start before the X display exists, otherwise the terminal dies
# a few seconds in and systemd restart-loops it (observed after a power cut:
# restart counter hit 10 with SIGTERM/143 every ~20s).
for _ in $(seq 1 30); do
    xdpyinfo -display :99 >/dev/null 2>&1 && break
    sleep 1
done

# The /portable flag is essential — without it MT5 ignores ini configs.
# /config: must point at a SPACE-FREE path; a path containing spaces gets
# mangled by the terminal ("cannot load config ... at start").
# exec so the terminal becomes the service MainPID (Type=simple tracks it).
exec wine "$MT5_PATH" /portable /config:C:\\start_ea.ini
SCRIPT
chmod +x "$REPO_DIR/scripts/start-mt5.sh"

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
    # Anchor to start-of-line: an unanchored "path = .*" also clobbers
    # model_path (and any other *_path key), which silently breaks ML loading.
    sed -i "s|^path = .*|path = $WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe|" \
        "$REPO_DIR/config/settings.ini" 2>/dev/null || warn "Could not rewrite settings.ini path — fix manually"
fi

# ──────────────────────────────────────────────
# Phase 8b: MT5 socket-server EA wiring
# ──────────────────────────────────────────────
# Wine's named pipes are broken, so the MetaTrader5 Python package can never
# reach the terminal. Instead an MQL5 EA runs inside the terminal and exposes
# the MT5 API over TCP 127.0.0.1:9000; bot/mt5_socket_client.py speaks to it.
log "Phase 8b: Wiring the MT5 socket-server EA"

# 1. EA source into MQL5/Experts/
mkdir -p "$MT5_DIR/MQL5/Experts"
cp "$REPO_DIR/scripts/mt5_socket_server.mq5" "$MT5_DIR/MQL5/Experts/mt5_socket_server.mq5"

# 2. Enable DLL imports (the EA calls ws2_32.dll). common.ini is UTF-16LE.
if [ -f "$MT5_DIR/Config/common.ini" ]; then
    cp "$MT5_DIR/Config/common.ini" "$MT5_DIR/Config/common.ini.bak"
    python3 - "$MT5_DIR/Config/common.ini" <<'PY' || warn "Could not patch AllowDllImport"
import sys
p = sys.argv[1]
d = open(p, "rb").read().decode("utf-16")
d = d.replace("AllowDllImport=0", "AllowDllImport=1")
open(p, "wb").write(d.encode("utf-16"))
PY
fi

# 3. Startup ini: broker login + EA auto-attach. Must be UTF-16 and live at a
#    space-free path. The chart symbol MUST exist on the broker after sync —
#    this broker uses .raw suffixes, so plain "EURUSD" silently never inits.
python3 - "$REPO_DIR/config/credentials.ini" "$WINEPREFIX/drive_c/start_ea.ini" <<'PY' || \
    warn "Could not write start_ea.ini — MT5 will not auto-login"
import configparser, sys
cred, out = sys.argv[1], sys.argv[2]
c = configparser.ConfigParser(strict=False)  # duplicate 'password' key in file
c.read(cred)
s = c["LOGIN"]
open(out, "wb").write((
    "[Common]\n"
    f"Login={s['account']}\n"
    f"Password={s['password']}\n"
    f"Server={s['server']}\n"
    "KeepPrivate=1\n"
    "NewsEnable=0\n"
    "\n"
    "[StartUp]\n"
    "Expert=mt5_socket_server\n"
    "Symbol=EURUSD.raw\n"
    "Period=H1\n"
).encode("utf-16"))
PY

# 4. Compile the EA headlessly. Note the capital E in MetaEditor64.exe, and
#    that it exits non-zero even on success — verify via compile.log.
( cd "$MT5_DIR" && DISPLAY=:99 timeout 200 wine MetaEditor64.exe \
    /compile:"MQL5\\Experts\\mt5_socket_server.mq5" \
    /log:"MQL5\\Experts\\compile.log" ) || true
if [ -f "$MT5_DIR/MQL5/Experts/mt5_socket_server.ex5" ]; then
    log "EA compiled: $(du -h "$MT5_DIR/MQL5/Experts/mt5_socket_server.ex5" | cut -f1)"
else
    warn "EA did not compile — check $MT5_DIR/MQL5/Experts/compile.log (UTF-16LE)"
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
systemctl --user enable doto-bot.service
systemctl --user enable doto-dashboard.service
systemctl --user enable doto-news.service

systemctl --user enable doto-backup.timer

log "Starting services (this will take ~60s)..."
systemctl --user start xvfb-mt5.service
sleep 3
systemctl --user start mt5.service
sleep 90  # MT5 under Wine needs ~85s before the socket EA binds port 9000
systemctl --user start doto-bot.service
systemctl --user start doto-dashboard.service
systemctl --user start doto-news.service

systemctl --user start doto-backup.timer

# ──────────────────────────────────────────────
# Phase 10: Verify deployment
# ──────────────────────────────────────────────
log "Phase 10: Verifying deployment"
echo ""
echo "=== Service Status ==="
for svc in xvfb-mt5 mt5 doto-bot doto-dashboard doto-news; do
    status=$(systemctl --user is-active "$svc" 2>/dev/null || echo "inactive")
    echo "  $svc: $status"
done

echo ""
echo "=== Timer Status ==="
for timer in doto-backup; do
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
