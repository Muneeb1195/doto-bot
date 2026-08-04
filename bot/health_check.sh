#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
BOT_LOG="$LOG_DIR/bot_$(date +%Y%m%d).log"

checks_passed=true

# 1. Check Xvfb
if ! pgrep -x Xvfb &>/dev/null; then
    echo "[health] FAIL: Xvfb not running"
    checks_passed=false
else
    echo "[health] OK: Xvfb running"
fi

# 2. Check MT5 terminal
if ! pgrep -f "terminal64.exe" &>/dev/null; then
    echo "[health] FAIL: MT5 terminal not running"
    checks_passed=false
else
    echo "[health] OK: MT5 terminal running"
fi

# 3. Check bot process
PID_FILE="$LOG_DIR/.bot_pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
    if [ -n "$PID" ] && [ "$PID" -gt 0 ] 2>/dev/null && kill -0 "$PID" 2>/dev/null; then
        echo "[health] OK: Bot PID $PID running"
    else
        echo "[health] FAIL: Bot PID $PID dead (stale PID file)"
        rm -f "$PID_FILE"
        checks_passed=false
    fi
else
    # Fallback: check main.py via ps
    if pgrep -f "main.py" &>/dev/null; then
        echo "[health] OK: Bot running (main.py found)"
    else
        echo "[health] FAIL: No bot process found"
        checks_passed=false
    fi
fi

# 4. Check recent log activity (skip if no log yet today)
if [ -f "$BOT_LOG" ]; then
    LAST_LINE=$(tail -1 "$BOT_LOG" 2>/dev/null || echo "")
    LAST_TS=$(echo "$LAST_LINE" | grep -oP '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | tail -1 || echo "")
    if [ -n "$LAST_TS" ]; then
        NOW=$(date +%s)
        LOG_SEC=$(date -d "$LAST_TS" +%s 2>/dev/null || echo 0)
        DIFF=$((NOW - LOG_SEC))
        if [ "$DIFF" -gt 600 ]; then
            echo "[health] WARN: Last log activity ${DIFF}s ago (>600s)"
        fi
    fi
fi

# 5. Check for orphaned Wine process groups (more than 1 winedevice = stale)
WINE_COUNT=$(pgrep -c winedevice.exe 2>/dev/null || echo 0)
if [ "$WINE_COUNT" -gt 4 ]; then
    echo "[health] WARN: $WINE_COUNT winedevice.exe instances (orphans?)"
fi

if [ "$checks_passed" = false ]; then
    echo "[health] UNHEALTHY — intervention needed"
    exit 1
fi

echo "[health] All checks passed"
exit 0
