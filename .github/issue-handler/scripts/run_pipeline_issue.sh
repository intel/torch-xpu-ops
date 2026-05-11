#!/bin/bash
# Run the orchestrator for a single issue.
# Usage: run_pipeline_issue.sh <issue_number>
set -euo pipefail

ISSUE=${1:?Usage: run_pipeline_issue.sh <issue_number>}
WORKDIR="$HOME/torch-xpu-ops/.github/issue-handler"
PYTHON="/mnt/miniforge3/envs/yucai3/bin/python"

cd "$WORKDIR"
set -a && source .env && set +a

echo "=== $(date -Iseconds) === Advancing issue #$ISSUE ==="

# Get current stage
BODY=$(GH_TOKEN=$REVIEW_GH_TOKEN gh api "repos/ZhaoqiongZ/torch-xpu-ops-exp/issues/$ISSUE" --jq '.body // ""' 2>&1)
STAGE=$(echo "$BODY" | grep -oP '(?<=<!-- agent:status:)\w+(?= -->)' || echo "NONE")
echo "Current stage: $STAGE"

# Terminal stages — nothing to do
case "$STAGE" in
    IN_REVIEW|PUBLIC_PR|CI_WATCH|MERGED|DONE|NEEDS_HUMAN)
        echo "Issue at terminal/blocked stage $STAGE — skipping"
        exit 0
        ;;
esac

# Run orchestrator
$PYTHON -m issue_handler.orchestrator --issue "$ISSUE" 2>&1

# Re-read stage after run
BODY_AFTER=$(GH_TOKEN=$REVIEW_GH_TOKEN gh api "repos/ZhaoqiongZ/torch-xpu-ops-exp/issues/$ISSUE" --jq '.body // ""' 2>&1)
STAGE_AFTER=$(echo "$BODY_AFTER" | grep -oP '(?<=<!-- agent:status:)\w+(?= -->)' || echo "NONE")
echo "Stage after run: $STAGE_AFTER"
echo "=== $(date -Iseconds) === Done ==="
