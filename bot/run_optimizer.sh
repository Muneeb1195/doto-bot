#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WINE_DIR="$PROJECT_DIR/wine"

export WINEPREFIX="$WINE_DIR"
export WINEARCH=win64

if grep -q "CHANGE_ME" "$PROJECT_DIR/config/credentials.ini"; then
    echo "ERROR: Update config/credentials.ini with your Doto account details first."
    exit 1
fi

MT5_EXE="$WINE_DIR/drive_c/Program Files/MetaTrader 5/terminal64.exe"

if ! pgrep -f "terminal64.exe" > /dev/null 2>&1; then
    echo "Starting MT5 terminal..."
    wine "$MT5_EXE" &
    echo "Waiting 30s for MT5 to initialize..."
    sleep 30
fi

echo "Running Auto-Optimizer (apply mode)..."
EXIT_CODE=0
wine "$WINE_DIR/drive_c/Program Files/Python312/python.exe" -u "$SCRIPT_DIR/auto_optimizer.py" --apply || EXIT_CODE=$?
echo "Auto-Optimizer finished (exit code $EXIT_CODE)"
