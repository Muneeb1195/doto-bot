#!/usr/bin/env bash
# Setup MT5 Socket Server EA inside the Wine MT5 terminal.
# Compiles the .mq5 file and places it in MQL5/Experts.
set -euo pipefail

export WINEPREFIX="${WINEPREFIX:-$HOME/.wine}"
export WINEARCH="${WINEARCH:-win64}"
export DISPLAY="${DISPLAY:-:99}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MQ5_SRC="$REPO_DIR/scripts/mt5_socket_server.mq5"
MT5_DIR="$WINEPREFIX/drive_c/Program Files/MetaTrader 5"
EXPERTS_DIR="$MT5_DIR/MQL5/Experts"
COMPILED_EX5="$EXPERTS_DIR/mt5_socket_server.ex5"

if [ ! -f "$MQ5_SRC" ]; then
    echo "ERROR: $MQ5_SRC not found"
    exit 1
fi

# Ensure MT5 is running
if ! pgrep -f terminal64.exe > /dev/null 2>&1; then
    echo "WARNING: MT5 terminal not running. Start it first with: systemctl --user start mt5.service"
    exit 1
fi

# Copy source to MT5 MQL5/Experts (so MetaEditor can find it)
cp "$MQ5_SRC" "$EXPERTS_DIR/"
echo "Copied $MQ5_SRC -> $EXPERTS_DIR/"

# Compile using MetaEditor
METAEDITOR="$MT5_DIR/metaeditor64.exe"
if [ -f "$METAEDITOR" ]; then
    echo "Compiling with MetaEditor..."
    wine "$METAEDITOR" /compile:"$EXPERTS_DIR/mt5_socket_server.mq5" 2>&1 | tail -5
    if [ -f "$COMPILED_EX5" ]; then
        echo "SUCCESS: $COMPILED_EX5"
    else
        echo "WARNING: Compilation may have failed. Check MetaEditor output."
    fi
else
    echo "WARNING: MetaEditor not found at $METAEDITOR"
    echo "Compile manually in MT5: File -> Open -> scripts/mt5_socket_server.mq5 -> Compile"
fi

echo ""
echo "Next steps:"
echo "1. In MT5 terminal, open Navigator (Ctrl+N)"
echo "2. Find 'mt5_socket_server' under Expert Advisors"
echo "3. Drag it onto any chart (or right-click -> Attach to chart)"
echo "4. Ensure 'AutoTrading' is enabled (click the AutoTrading button in toolbar)"
echo "5. The EA will start listening on port 9000"
