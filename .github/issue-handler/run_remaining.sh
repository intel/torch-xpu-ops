#!/bin/bash
# Run orchestrator on issues that still need work
set -a && source /home/stonepia/torch-xpu-ops/.github/issue-handler/.env && set +a
PYTHON=~/pytorch/.venv/bin/python
DIR=~/torch-xpu-ops/.github/issue-handler

# Only issues that still need code_fix work:
# - #1963: IN_REVIEW but needs re-verify (existing fix)
# - #2253: IMPLEMENTING, agent was killed mid-attempt
# - #2712: IMPLEMENTING, git checkout crashed (now fixed)
ISSUES=(2253 2712)

echo "=== Starting pipeline run on ${#ISSUES[@]} issues ==="
echo "$(date)"

for issue in "${ISSUES[@]}"; do
    echo ""
    echo "================================================================"
    echo "=== Processing #${issue} at $(date) ==="
    echo "================================================================"
    $PYTHON -m issue_handler.orchestrator --issue "$issue" 2>&1
    rc=$?
    echo "=== #${issue} finished with rc=${rc} at $(date) ==="
done

echo ""
echo "=== All done at $(date) ==="

# Print final status
echo ""
echo "=== Final Status ==="
$PYTHON -c "
from issue_handler.utils.git import get_issue_detail
from issue_handler.utils.body_templates import get_status
for n in [1951, 1963, 2015, 2253, 2295, 2512, 2518, 2609, 2615, 2712, 2800, 2891, 2953]:
    detail = get_issue_detail('intel/torch-xpu-ops', n)
    body = detail.get('body', '') or ''
    status = get_status(body)
    print(f'#{n}: {status}')
"
