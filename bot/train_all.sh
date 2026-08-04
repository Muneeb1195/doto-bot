#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WINE="$PROJECT_DIR/wine/drive_c/Program Files/Python312/python.exe"
WINEPREFIX="$PROJECT_DIR/wine"
LOG="$PROJECT_DIR/logs/ml_training.log"

export WINEPREFIX WINEARCH=win64

: > "$LOG"
echo "ML Training started $(date)" >> "$LOG"

train() {
    local sym=$1 sl=$2 tp=$3
    echo "=== Training $sym (SL=$sl TP=$tp) ===" >> "$LOG"
    date >> "$LOG"
    wine "$WINE" -u "$SCRIPT_DIR/train_model.py" \
        --symbols "$sym" --sl-atr "$sl" --tp-atr "$tp" \
        --max-hold 12 --years 2 --prune 2>&1 | tee -a "$LOG"
    echo "=== Done $sym ===" >> "$LOG"
    local model="$PROJECT_DIR/models/model_${sym//./_}.pkl"
    if [ -f "$model" ]; then
        echo "  Model saved: $(du -h "$model" | cut -f1)" >> "$LOG"
    fi
}

# Train all new symbols
train XAU500.raw 1.5 2.5
train EURJPY.raw 1.5 2.0
train NZDUSD.raw 1.5 2.0
train USDJPY.raw 1.5 2.5
train GBPJPY.raw 1.5 2.5
train US500.raw 1.5 2.0
train LTCUSD.raw 1.0 2.0
train DOGUSD.raw 1.0 3.0

echo "=== ALL DONE $(date) ===" >> "$LOG"
