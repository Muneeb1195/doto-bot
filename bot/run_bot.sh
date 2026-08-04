#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WINE_DIR="$PROJECT_DIR/wine"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/.bot_pid"

export WINEPREFIX="$WINE_DIR"
export WINEARCH=win64

cleanup() {
    echo "[run_bot] Cleaning up stale Wine/MT5 processes..."
    killall -9 terminal64.exe metaeditor64.exe winedevice.exe wineserver 2>/dev/null || true
    killall -9 services.exe plugplay.exe explorer.exe 2>/dev/null || true
    sleep 1
    rm -f "$WINE_DIR/.lock" "$WINE_DIR/.update-lock" 2>/dev/null || true
    rm -f /tmp/.wine-* 2>/dev/null || true
}

ensure_xvfb() {
    if ! pgrep -x Xvfb &>/dev/null; then
        echo "[run_bot] Xvfb not running — starting..."
        Xvfb :99 -screen 0 1280x720x16 -nolisten tcp &>/dev/null &
        sleep 2
        if pgrep -x Xvfb &>/dev/null; then
            echo "[run_bot] Xvfb started (PID $(pgrep -x Xvfb))"
        else
            echo "[run_bot] WARNING: Xvfb failed to start"
        fi
    else
        echo "[run_bot] Xvfb already running (PID $(pgrep -x Xvfb))"
    fi
    export DISPLAY=:99
}

start_terminal() {
    echo "[run_bot] Starting MetaTrader 5 terminal..."
    wine "$WINE_DIR/drive_c/Program Files/MetaTrader 5/terminal64.exe" &>/dev/null &
    TERM_PID=$!
    # Wait up to 30s for MT5 to be connectable
    for i in $(seq 1 6); do
        sleep 5
        if winedbg --command "info wnd" 2>/dev/null | grep -qi "MetaTrader"; then
            echo "[run_bot] MT5 terminal window detected"
            return 0
        fi
    done
    echo "[run_bot] MT5 terminal may not be ready yet, continuing..."
}

start_bot() {
    echo "[run_bot] Starting TrendBot..."
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$OLD_PID" ] && [ "$OLD_PID" -gt 0 ] 2>/dev/null && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "[run_bot] Bot already running (PID $OLD_PID), exiting"
            exit 1
        fi
        rm -f "$PID_FILE"
    fi
    echo $$ > "$PID_FILE"
    wine "$WINE_DIR/drive_c/Program Files/Python312/python.exe" -u "$SCRIPT_DIR/main.py"
    BOT_EXIT=$?
    rm -f "$PID_FILE"
    echo "[run_bot] TrendBot exited with code $BOT_EXIT"
    return $BOT_EXIT
}

if [ ! -f "$PROJECT_DIR/config/credentials.ini" ]; then
    echo "ERROR: config/credentials.ini not found — create it from .env.example or set env vars."
    exit 1
fi
if grep -q "CHANGE_ME" "$PROJECT_DIR/config/credentials.ini"; then
    echo "ERROR: Update config/credentials.ini with your Doto account details first."
    exit 1
fi

cleanup
ensure_xvfb
start_terminal

# Main restart loop
MAX_RESTART_DELAY=60
RESTART_DELAY=5
while true; do
    start_bot
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 130 ]; then
        echo "[run_bot] Clean exit (code $EXIT_CODE) — not restarting"
        break
    fi
    echo "[run_bot] Restarting in ${RESTART_DELAY}s (exit code $EXIT_CODE)..."
    sleep "$RESTART_DELAY"
    RESTART_DELAY=$((RESTART_DELAY * 2))
    if [ "$RESTART_DELAY" -gt "$MAX_RESTART_DELAY" ]; then
        RESTART_DELAY="$MAX_RESTART_DELAY"
    fi
    # Re-check Xvfb before each restart attempt
    ensure_xvfb
    # Re-check MT5 every 3rd restart
    if ! pgrep -f "terminal64.exe" &>/dev/null; then
        cleanup
        start_terminal
    fi
done
