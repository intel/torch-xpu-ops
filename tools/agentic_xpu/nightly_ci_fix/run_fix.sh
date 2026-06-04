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

# Nightly CI UT fix pipeline for XPU.
#
# Usage:
#   bash run_fix.sh <report_file> [DATE]
#
# Arguments:
#   report_file   Path to the nightly CI failure report (plain text or markdown).
#                 Defaults to agent_inputs/nightly_CI_ut_fix/report_<DATE>.md if not given.
#   DATE          Date string used for output naming, e.g. 0521 or 2026-05-21.
#                 Defaults to today (MMDD format).
#
# Outputs:
#   agent_outputs/nightly_CI_ut_fix/summary_<DATE>.md
#
# Env vars (optional overrides):
#   OPENCODE_BIN   path to opencode binary (default: opencode in PATH)
#   WORKSPACE      repo working directory (default: git root of this repo)
#   ENV_FILE       path to .env (default: tools/agentic_xpu/.env)
#   BUILD_ENV      path to build_pytorch.env (default: <repo_root>/build_pytorch.env)
#   PYTORCH_DIR    path to local pytorch source (default: ~/pytorch)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

# --- Parse args ---
REPORT_FILE="${1:-}"
DATE_ARG="${2:-$(date '+%m%d')}"

# --- Resolve paths ---
OPENCODE_BIN="${OPENCODE_BIN:-$(command -v opencode 2>/dev/null || echo "")}"
[[ -n "$OPENCODE_BIN" ]] || { echo "ERROR: opencode not found. Set OPENCODE_BIN or add to PATH." >&2; exit 1; }

WORKSPACE="${WORKSPACE:-$REPO_ROOT}"

# Load shared env (tokens, build config, PYTORCH_DIR, etc.)
ENV_FILE="${ENV_FILE:-$REPO_ROOT/tools/agentic_xpu/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

BUILD_ENV="${BUILD_ENV:-$REPO_ROOT/build_pytorch.env}"
PYTORCH_DIR="${PYTORCH_DIR:-$HOME/pytorch}"

# Default report path
if [[ -z "$REPORT_FILE" ]]; then
  REPORT_FILE="$REPO_ROOT/agent_inputs/nightly_CI_ut_fix/report_${DATE_ARG}.md"
fi

[[ -f "$REPORT_FILE" ]] || {
  echo "ERROR: report file not found: $REPORT_FILE" >&2
  echo "Usage: bash run_fix.sh <report_file> [DATE]" >&2
  exit 1
}

# --- Output path ---
OUTPUT_DIR="$REPO_ROOT/agent_outputs/nightly_CI_ut_fix"
OUTPUT_FILE="$OUTPUT_DIR/summary_${DATE_ARG}.md"
mkdir -p "$OUTPUT_DIR"

# --- Sync skill into workspace ---
SKILL_SRC="$REPO_ROOT/.github/skills/xpu-nightly-ci-fix"
SKILL_DST="$WORKSPACE/.opencode/skills/xpu-nightly-ci-fix"
mkdir -p "$SKILL_DST"
cp "$SKILL_SRC/SKILL.md" "$SKILL_DST/SKILL.md"

# --- Run log ---
LOG_FILE="$OUTPUT_DIR/run_${DATE_ARG}.log"

echo "=== XPU Nightly CI UT Fix: $DATE_ARG  ($(date -Iseconds)) ==="
echo "  Report:   $REPORT_FILE"
echo "  Output:   $OUTPUT_FILE"
echo "  Log:      $LOG_FILE"
echo "  PyTorch:  $PYTORCH_DIR"

REPORT_CONTENT="$(cat "$REPORT_FILE")"

PROMPT="Use the xpu-nightly-ci-fix skill.

CI failure report for $DATE_ARG:

$REPORT_CONTENT

Work in pytorch source directory: $PYTORCH_DIR
Build env: $BUILD_ENV

Run all steps (Step 1–6) from the skill.

At the end (Step 6), write the summary report to:
  $OUTPUT_FILE

Use the exact format from the skill: Report Info section, one entry per failing test with
root cause / fix / AR, a Summary Table, and a Verification section."

"$OPENCODE_BIN" run \
  --dir "$PYTORCH_DIR" \
  --dangerously-skip-permissions \
  --title "nightly-ci-fix-$DATE_ARG" \
  "$PROMPT" \
  2>&1 | tee "$LOG_FILE"

rc=${PIPESTATUS[0]}
echo "=== Finished: $(date -Iseconds) (exit $rc) ==="

if [[ -f "$OUTPUT_FILE" ]]; then
  echo "Summary written: $OUTPUT_FILE"
else
  echo "WARNING: summary file not produced at $OUTPUT_FILE" >&2
  rc=1
fi

exit "$rc"
