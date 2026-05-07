#!/usr/bin/env bash
set -euo pipefail

# Audit script: validates that a scan report + ledger are consistent.
#
# Usage:
#   bash scripts/audit_scan_report.sh <scan-file> [ledger-file]
#
# Exit codes:
#   0 - PASSED (no pending rows, report format valid)
#   1 - actionable candidates remain pending
#   2 - input error (missing files, missing tools)

scan_file="${1:?usage: audit_scan_report.sh <scan-file> [ledger-file]}"
ledger_file="${2:-artifacts/candidate_ledger.jsonl}"

# Check prerequisites
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required but not installed." >&2
  exit 2
fi

[[ -f "$scan_file" ]] || { echo "ERROR: scan file not found: $scan_file" >&2; exit 2; }
[[ -f "$ledger_file" ]] || { echo "ERROR: ledger file not found: $ledger_file" >&2; exit 2; }

# Validate JSONL format (single jq pass — fails on first malformed line)
if ! jq -e '.' "$ledger_file" > /dev/null 2>&1; then
  # Find the offending line for a useful error message
  line_num=0
  while IFS= read -r line; do
    line_num=$((line_num + 1))
    if ! echo "$line" | jq -e '.' >/dev/null 2>&1; then
      echo "ERROR: malformed JSON at $ledger_file line $line_num" >&2
      exit 2
    fi
  done < "$ledger_file"
fi

# Count pending actionable rows
pending="$(jq -s '[.[] | select(.title_status == "pass" and .deep_status != "reject" and .local_status == "pending")] | length' "$ledger_file")"

# Count deep-rejected rows that are still pending (should be 0 — deep-reject implies done)
deep_rejected_pending="$(jq -s '[.[] | select(.title_status == "pass" and .deep_status == "reject" and .local_status == "pending")] | length' "$ledger_file")"

# Count report entries
rows="$(awk '/^[0-9]+\. `#/ { entries += 1 } END { print entries + 0 }' "$scan_file")"

# Count ledger done rows
done_count="$(jq -s '[.[] | select(.local_status == "done" and .title_status == "pass")] | length' "$ledger_file")"

echo "ledger pending=$pending"
echo "ledger deep_rejected_pending=$deep_rejected_pending"
echo "ledger done=$done_count"
echo "report rows=$rows"

# Warn on divergence (not fatal — deep-rejected rows are done but excluded from report)
if [[ "$rows" -gt 0 ]] && [[ "$done_count" -gt 0 ]]; then
  if [[ "$rows" -ne "$done_count" ]]; then
    echo "NOTE: report rows ($rows) != ledger done ($done_count) — expected if deep-rejected rows exist" >&2
  fi
fi

# Check pending
if [[ "$pending" -gt 0 ]]; then
  echo "INCOMPLETE: $pending actionable rows remain pending" >&2
  exit 1
fi

if [[ "$deep_rejected_pending" -gt 0 ]]; then
  echo "WARNING: $deep_rejected_pending deep-rejected rows still have local_status=pending (should be done)" >&2
fi

# Validate report format
if grep -qE '^[0-9]+\. `#' "$scan_file"; then
  if ! grep -q '^Local XPU result: `' "$scan_file"; then
    echo "ERROR: report has numbered entries but missing 'Local XPU result:' lines" >&2
    exit 1
  fi
fi

echo "PASSED"
