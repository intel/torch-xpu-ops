#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
LOCKFILE="/tmp/pytorch-agent-pipeline.lock"

# Prevent overlapping runs
exec 200>"$LOCKFILE"
flock -n 200 || { echo "$(date) Pipeline already running, skipping"; exit 0; }

cd "$AGENT_DIR"
source .env

# Pass all arguments through (e.g., --issue 3509)
python3 scripts/run_pipeline.py "$@" 2>&1
