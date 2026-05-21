#!/usr/bin/env bash
# pytorch-agent cron wrapper
# Runs one discovery+advance cycle, logs output
# Install: crontab -e → */5 * * * * ~/.github/pytorch-agent/scripts/cron.sh

set -euo pipefail

AGENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$AGENT_DIR/logs"
mkdir -p "$LOG_DIR"

# Source env if exists
[[ -f "$AGENT_DIR/.env" ]] && set -a && source "$AGENT_DIR/.env" && set +a

# Lock — skip if previous cycle still running
LOCKFILE="$LOG_DIR/cron.lock"
if [[ -f "$LOCKFILE" ]]; then
    PID=$(cat "$LOCKFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "$(date -Iseconds) Previous cycle still running (PID $PID), skipping" >> "$LOG_DIR/cron.log"
        exit 0
    fi
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# Run one cycle
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
echo "$(date -Iseconds) Starting cycle" >> "$LOG_DIR/cron.log"

cd "$AGENT_DIR"
python3 scripts/run_pipeline.py --once >> "$LOG_DIR/cycle-$TIMESTAMP.log" 2>&1
EXIT_CODE=$?

echo "$(date -Iseconds) Cycle finished (exit=$EXIT_CODE)" >> "$LOG_DIR/cron.log"

# Rotate: keep last 100 cycle logs
ls -1t "$LOG_DIR"/cycle-*.log 2>/dev/null | tail -n +101 | xargs -r rm
