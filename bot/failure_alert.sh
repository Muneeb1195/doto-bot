#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${1:-unknown}"

# Ignore timeouts — Wine processes don't respond to SIGTERM gracefully
if [ "${SERVICE_RESULT:-}" = "timeout" ] || [ "${EXIT_CODE:-}" = "killed" ]; then
    exit 0
fi

PROJECT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
CREDENTIALS="$PROJECT_DIR/config/credentials.ini"

if [ -f "$CREDENTIALS" ]; then
    WEBHOOK=$(grep -oP '(?<=^discord_url = ).*' "$CREDENTIALS" 2>/dev/null || echo "")
fi

HOSTNAME=$(hostname)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
MSG="{\"embeds\":[{\"title\":\"Service Failure Alert\",\"color\":16711680,\"fields\":[{\"name\":\"Service\",\"value\":\"$SERVICE_NAME\",\"inline\":true},{\"name\":\"Host\",\"value\":\"$HOSTNAME\",\"inline\":true},{\"name\":\"Time\",\"value\":\"$TIMESTAMP\"}]}]}"

if [ -n "$WEBHOOK" ]; then
    curl -s -H "Content-Type: application/json" -d "$MSG" "$WEBHOOK" >/dev/null 2>&1 || true
fi

# Optional healthchecks.io ping
if [ -f "$CREDENTIALS" ]; then
    HC_URL=$(grep -oP '(?<=^healthchecks_url = ).*' "$CREDENTIALS" 2>/dev/null || echo "")
    if [ -n "$HC_URL" ]; then
        curl -fsS -m 10 --retry 3 "$HC_URL/fail" >/dev/null 2>&1 || true
    fi
fi
