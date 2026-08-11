#!/usr/bin/env bash
# service-ctl.sh — Unified control script for Doto MT5 Bot services.
#
# Usage:
#   bash scripts/service-ctl.sh start     # start all services in order
#   bash scripts/service-ctl.sh stop      # stop all services in reverse order
#   bash scripts/service-ctl.sh restart   # stop then start
#   bash scripts/service-ctl.sh status    # show status of all services
#   bash scripts/service-ctl.sh health    # deep health check
#
# Start order:  xvfb-mt5 → xfce4-mt5 → x11vnc-mt5 → mt5 → mt5server → doto-bot → doto-dashboard → doto-news
# Stop order:   reverse of start order.

set -euo pipefail

SERVICES=(xvfb-mt5 xfce4-mt5 x11vnc-mt5 mt5 mt5server doto-bot doto-dashboard doto-news)
TIMER_SERVICES=(mt5-watchdog)
LOG_FILE="${LOG_FILE:-$HOME/doto-mt5-bot/logs/bot.log}"

cmd_to_service() {
    case "$1" in
        mt5-watchdog) echo "mt5-watchdog.timer" ;;
        *) echo "$1.service" ;;
    esac
}

do_start() {
    for svc in "${SERVICES[@]}" "${TIMER_SERVICES[@]}"; do
        unit="$(cmd_to_service "$svc")"
        echo "  Starting $unit..."
        systemctl --user start "$unit" 2>/dev/null || echo "    (already running or failed)"
    done
    echo "All services started."
}

do_stop() {
    for ((i=${#TIMER_SERVICES[@]}-1; i>=0; i--)); do
        svc="${TIMER_SERVICES[$i]}"
        unit="$(cmd_to_service "$svc")"
        echo "  Stopping $unit..."
        systemctl --user stop "$unit" 2>/dev/null || true
    done
    for ((i=${#SERVICES[@]}-1; i>=0; i--)); do
        svc="${SERVICES[$i]}"
        unit="$(cmd_to_service "$svc")"
        echo "  Stopping $unit..."
        systemctl --user stop "$unit" 2>/dev/null || true
    done
    echo "All services stopped."
}

do_status() {
    printf "%-20s %-10s %-10s %s\n" "SERVICE" "ACTIVE" "SUB" "SINCE"
    printf "%-20s %-10s %-10s %s\n" "-------" "------" "-----" "-----"
    for svc in "${SERVICES[@]}" "${TIMER_SERVICES[@]}"; do
        unit="$(cmd_to_service "$svc")"
        state=$(systemctl --user is-active "$unit" 2>/dev/null || echo "unknown")
        sub=$(systemctl --user show -p SubState --value "$unit" 2>/dev/null || echo "-")
        since=$(systemctl --user show -p ActiveEnterTimestamp --value "$unit" 2>/dev/null || echo "-")
        printf "%-20s %-10s %-10s %s\n" "$svc" "$state" "$sub" "$since"
    done
}

do_health() {
    echo "=== Doto MT5 Bot Health Check ==="
    local ok=0 fail=0

    # 1. Bot log: "Bot state loaded" marker
    if [ -f "$LOG_FILE" ]; then
        if grep -q "Bot state loaded" "$LOG_FILE" 2>/dev/null; then
            echo "  [OK] Bot state loaded marker present in bot.log"
            ((ok++))
        else
            echo "  [WARN] No \'Bot state loaded\' marker in bot.log (may still be starting)"
            ((fail++))
        fi
    else
        echo "  [WARN] bot.log not found at $LOG_FILE"
        ((fail++))
    fi

    # 2. mt5server RPyC port
    if ss -tlnp 2>/dev/null | grep -q 18812; then
        echo "  [OK] mt5server listening on port 18812"
        ((ok++))
    else
        echo "  [FAIL] mt5server NOT listening on port 18812"
        ((fail++))
    fi

    # 3. MT5 terminal process
    if pgrep -x terminal64.exe >/dev/null 2>&1; then
        echo "  [OK] terminal64.exe running"
        ((ok++))
    else
        echo "  [FAIL] terminal64.exe NOT running"
        ((fail++))
    fi

    # 4. Dashboard HTTP (expect 401 with auth)
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8501/ 2>/dev/null || echo "000")
    if [ "$http_code" = "401" ]; then
        echo "  [OK] Dashboard responding (HTTP 401 - auth enabled)"
        ((ok++))
    elif [ "$http_code" = "200" ]; then
        echo "  [WARN] Dashboard responding (HTTP 200 - no auth?)"
        ((ok++))
    elif [ "$http_code" != "000" ]; then
        echo "  [WARN] Dashboard HTTP $http_code"
        ((fail++))
    else
        echo "  [FAIL] Dashboard not responding on port 8501"
        ((fail++))
    fi

    # 5. RPyC ping (Python one-liner)
    if command -v python3 >/dev/null; then
        local ping_result
        ping_result=$(python3 -c "
import socket, sys
try:
    s = socket.socket(); s.settimeout(5)
    s.connect(('127.0.0.1', 18812))
    s.close(); print('OK')
except Exception:
    print('FAIL')
" 2>/dev/null || echo "FAIL")
        if [ "$ping_result" = "OK" ]; then
            echo "  [OK] RPyC port 18812 connectable"
            ((ok++))
        else
            echo "  [FAIL] RPyC port 18812 not connectable"
            ((fail++))
        fi
    fi

    echo ""
    echo "Result: $ok passed, $fail warnings/failures"
    [ "$fail" -eq 0 ]
}

# ──────────────────────────────────────────────
case "${1:-}" in
    start)    do_start ;;
    stop)     do_stop ;;
    restart)  do_stop; do_start ;;
    status)   do_status ;;
    health)   do_health ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|health}"
        exit 1
        ;;
esac
