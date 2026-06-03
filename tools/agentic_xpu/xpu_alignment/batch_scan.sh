#!/usr/bin/env bash
# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

set -euo pipefail

# Batch XPU alignment scan (arbitrary date range, with resume support).
# Usage: batch_scan.sh <start> <end>  OR  batch_scan.sh <N>d
# Env:   OPENCODE_BIN  WORKSPACE  ENV_FILE  GH_TOKEN  GITHUB_TOKEN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY_DIR="$SCRIPT_DIR"

# --- Parse date args ---
if [[ $# -lt 1 ]]; then
  echo "Usage: batch_scan.sh <start> <end>  OR  batch_scan.sh <N>d" >&2; exit 1
fi
if [[ "$1" =~ ^([0-9]+)d$ ]]; then
  DAYS="${BASH_REMATCH[1]}"
  END_DATE="$(date -d 'yesterday' '+%Y-%m-%d' 2>/dev/null || date -v-1d '+%Y-%m-%d')"
  START_DATE="$(date -d "$END_DATE - ${DAYS} days" '+%Y-%m-%d' 2>/dev/null || date -v-${DAYS}d '+%Y-%m-%d')"
elif [[ $# -ge 2 ]]; then
  START_DATE="$1"; END_DATE="$2"
else
  echo "Usage: batch_scan.sh <start> <end>  OR  batch_scan.sh <N>d" >&2; exit 1
fi

RUN_NAME="${START_DATE}_to_${END_DATE}"
RUN_DIR="$ENTRY_DIR/runs/$RUN_NAME"
SESSION_FILE="$RUN_DIR/.session_id"
mkdir -p "$RUN_DIR"

# shellcheck source=common.sh
. "$ENTRY_DIR/common.sh"

echo "=== XPU Alignment Batch Scan: $START_DATE → $END_DATE  ($(date -Iseconds)) ==="

DAYS_COUNT=$(( ( $(date -d "$END_DATE" +%s 2>/dev/null || date -j -f '%Y-%m-%d' "$END_DATE" +%s) \
              - $(date -d "$START_DATE" +%s 2>/dev/null || date -j -f '%Y-%m-%d' "$START_DATE" +%s) ) / 86400 ))

PROMPT="Use the pytorch-cuda-fix-xpu-alignment skill. \
Scan pytorch/pytorch for issues, PRs, and commits between $START_DATE and $END_DATE \
(window: ${START_DATE}T00:00:00Z to ${END_DATE}T23:59:59Z). \
Work in: $RUN_DIR. Run all steps (Step 0-3). Zero pending rows. \
XPU interpreter: $XPU_PYTHON. \
Large window (~${DAYS_COUNT} days): paginate ALL candidates, split date ranges at the 1000-result cap. \
Resume from ledger if interrupted."

OPENCODE_ARGS=(--dir "$WORKSPACE" --dangerously-skip-permissions --title "XPU scan $RUN_NAME")

if [[ -f "$SESSION_FILE" ]]; then
  echo "Resuming session: $(cat "$SESSION_FILE")"
  OPENCODE_ARGS+=(--session "$(cat "$SESSION_FILE")")
  PROMPT="Continue the pytorch-cuda-fix-xpu-alignment scan. \
Dir: $RUN_DIR. Window: $START_DATE to $END_DATE. \
Resume from ledger. Complete until zero pending rows and audit passes."
fi

"$OPENCODE_BIN" run "${OPENCODE_ARGS[@]}" "$PROMPT" \
  2>&1 | tee -a "$RUN_DIR/run.log"

rc=${PIPESTATUS[0]}

# Save session ID for resume
if [[ ! -f "$SESSION_FILE" ]]; then
  sid="$(grep -oE 'session[: ]+[a-f0-9-]+' "$RUN_DIR/run.log" 2>/dev/null \
       | grep -oE '[a-f0-9-]{8,}' | tail -1 || true)"
  [[ -n "$sid" ]] && echo "$sid" > "$SESSION_FILE"
fi

echo "=== Finished: $(date -Iseconds) (exit $rc) ==="
run_post_scan "$RUN_DIR" "$rc"
exit $?
