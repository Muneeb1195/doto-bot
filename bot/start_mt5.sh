#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export WINEPREFIX="$PROJECT_DIR/wine"
export WINEARCH=win64

if command -v Xvfb &>/dev/null; then
    Xvfb :99 -screen 0 1280x720x16 -nolisten tcp &>/dev/null &
    export DISPLAY=:99
    echo "Headless mode via Xvfb"
    sleep 1
fi

MT5_EXE="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"
if [ ! -f "$MT5_EXE" ]; then
    echo "Error: MT5 not found at $MT5_EXE"
    exit 1
fi

exec wine "$MT5_EXE"
