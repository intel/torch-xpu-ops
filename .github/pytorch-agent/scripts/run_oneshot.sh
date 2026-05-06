#!/usr/bin/env bash
# Ad-hoc single-issue run.
# Usage: ./scripts/run_oneshot.sh --issue 3509
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$AGENT_DIR"
set -a && source .env && set +a

python3 scripts/run_pipeline.py "$@"
