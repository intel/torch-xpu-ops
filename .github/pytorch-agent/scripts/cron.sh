#!/usr/bin/env bash
# Cron entry point — runs one pipeline cycle.
# Scheduled via: */15 * * * * /path/to/cron.sh >> /path/to/logs/cron.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env (bare KEY=value format exported into environment)
[[ -f "$AGENT_DIR/.env" ]] && set -a && source "$AGENT_DIR/.env" && set +a

exec python3 "$AGENT_DIR/scripts/run_pipeline.py" --once "$@"
