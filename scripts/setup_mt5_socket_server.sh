#!/usr/bin/env bash
# Setup MT5 Socket Server EA inside the Wine MT5 terminal.
# Places the .mq5 file in MQL5/Experts for manual compilation.
set -euo pipefail

export WINEPREFIX="${WINEPREFIX:-$HOME/.wine}"
export WINEARCH="${WINEARCH:-win64}"
export DISPLAY="${DISPLAY:-:99}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MQ5_SRC="$REPO_DIR/scripts/mt5_socket_server.mq5"
MT5_DIR="$WINEPREFIX/drive_c/Program Files/MetaTrader 5"
EXPERTS_DIR="$MT5_DIR/MQL5/Experts"

if [ ! -f "$MQ5_SRC" ]; then
    echo "ERROR: $MQ5_SRC not found"
    exit 1
fi

# Copy source to MT5 MQL5/Experts (so MetaEditor can find it)
cp "$MQ5_SRC" "$EXPERTS_DIR/"
echo "Copied $MQ5_SRC -> $EXPERTS_DIR/"

echo ""
echo "=== MANUAL COMPILATION REQUIRED ==="
echo "MetaEditor CLI compilation is not supported in this Wine setup."
echo "Compile manually:"
echo "1. Open MetaEditor in MT5 (F4 key or Tools -> MetaQuotes Language Editor)"
echo "2. File -> Open -> select 'mt5_socket_server.mq5' in MQL5/Experts"
echo "3. Press F7 or click 'Compile' button"
echo "4. Check for 0 errors in the bottom panel"
echo ""
echo "=== AFTER COMPILATION ==="
echo "1. In MT5 terminal, open Navigator (Ctrl+N)"
echo "2. Find 'mt5_socket_server' under Expert Advisors"
echo "3. Drag it onto any chart (or right-click -> Attach to chart)"
echo "4. Ensure 'AutoTrading' is enabled (click the AutoTrading button)"
echo "5. Check the 'Experts' tab for 'MT5Socket' log messages"
echo "6. The EA listens on port 9000"
echo ""
echo "=== VERIFY ==="
echo "From Linux: .venv/bin/python -c \"from mt5_socket_client import MT5SocketClient; c=MT5SocketClient(); c.connect(); print(c._call('PING')); c.disconnect()\""
