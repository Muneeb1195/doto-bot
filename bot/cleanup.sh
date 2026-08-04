#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Killing MT5 processes..."
pkill -f "terminal64.exe" 2>/dev/null || true
pkill -f "metaeditor64.exe" 2>/dev/null || true
pkill -f "winedevice.exe" 2>/dev/null || true
sleep 1
pkill -f "wineserver" 2>/dev/null || true
sleep 1

echo "Removing Wine locks..."
rm -f "$PROJECT_DIR/wine/.lock" "$PROJECT_DIR/wine/.update-lock" 2>/dev/null || true
rm -f /tmp/.wine-* 2>/dev/null || true

echo "Cleanup complete."
