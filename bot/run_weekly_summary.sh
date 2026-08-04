#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
export WINEPREFIX="$DIR/wine"
PYTHON="$DIR/wine/drive_c/Program Files/Python312/python.exe"

WEBHOOK="${DISCORD_WEBHOOK_URL:-$(grep -oP '(?<=discord_url = ).*' "$DIR/config/credentials.ini" 2>/dev/null || echo "")}"
if [ -z "$WEBHOOK" ]; then
  echo "No discord_url in credentials.ini"
  exit 1
fi

"$PYTHON" -u "$DIR/bot/weekly_summary.py" "$WEBHOOK"
