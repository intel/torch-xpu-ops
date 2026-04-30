#!/bin/bash
set -euo pipefail
LOCKFILE="/tmp/pytorch-agent-pipeline.lock"

# Prevent overlapping runs
exec 200>"$LOCKFILE"
flock -n 200 || { echo "$(date) Pipeline already running, skipping"; exit 0; }

cd ~/torch-xpu-ops/.github/pytorch-agent
source .env
python3 scripts/run_pipeline.py --issue 3509 2>&1
