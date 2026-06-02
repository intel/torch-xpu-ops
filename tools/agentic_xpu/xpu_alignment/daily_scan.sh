#!/usr/bin/env bash
set -euo pipefail

# Daily XPU alignment scan.
# Usage: bash daily_scan.sh [YYYY-MM-DD]
# Env:   OPENCODE_BIN  WORKSPACE  ENV_FILE  GH_TOKEN  GITHUB_TOKEN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY_DIR="$SCRIPT_DIR"

SCAN_DATE="${1:-$(date -d 'yesterday' '+%Y-%m-%d' 2>/dev/null || date -v-1d '+%Y-%m-%d')}"
RUN_DIR="$ENTRY_DIR/runs/$SCAN_DATE"
mkdir -p "$RUN_DIR"

# shellcheck source=common.sh
. "$ENTRY_DIR/common.sh"

echo "=== XPU Alignment Daily Scan: $SCAN_DATE  ($(date -Iseconds)) ==="

PROMPT="Use the pytorch-cuda-fix-xpu-alignment skill. \
Scan pytorch/pytorch for issues, PRs, and bug-fix commits from $SCAN_DATE \
(window: ${SCAN_DATE}T00:00:00Z to ${SCAN_DATE}T23:59:59Z). \
Work in: $RUN_DIR. Run all steps (Step 0-3). Zero pending rows. \
XPU interpreter: $XPU_PYTHON"

"$OPENCODE_BIN" run \
  --dir "$WORKSPACE" \
  --dangerously-skip-permissions \
  --title "XPU scan $SCAN_DATE" \
  "$PROMPT" \
  2>&1 | tee -a "$RUN_DIR/run.log"

rc=${PIPESTATUS[0]}
echo "=== Finished: $(date -Iseconds) (exit $rc) ==="
run_post_scan "$RUN_DIR" "$rc"
exit $?
