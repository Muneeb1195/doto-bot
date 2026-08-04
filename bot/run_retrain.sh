#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
WINEPREFIX="$PROJECT_DIR/wine" "$PROJECT_DIR/wine/drive_c/Program Files/Python312/python.exe" \
  -u "$SCRIPT_DIR/train_model.py" --retrain-all --tune --years=3 2>&1
echo "Retrain complete at $(date)"
