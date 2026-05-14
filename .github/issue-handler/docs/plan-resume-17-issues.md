# Resume Plan: 17 Upstream Issues Pipeline

## What Was Done This Session

### Code Fixes Applied (on branch `agent/pipeline-redesign-v2`)

1. **verify_existence.py — xfail/skip detection**
   - `_run_test()` now checks for `\d+ xfailed` in output → returns False (bug still exists)
   - Checks for all-skipped (pytest: `\d+ skipped` with no `\d+ passed`)
   - Checks for unittest-style skip: `OK (skipped=N)` and `Ran N test` where N == skipped count
   - Removed `[:5]` limit on test list — runs ALL tests
   - Changed `-xvs` to `-v` so all tests run even if one fails

2. **triage_agent.py — visible Target Repository**
   - Added `update_section(new_body, "Target Repository", target_repo)` before `append_log`
   - Added `**Target Repository:** \`{target_repo}\`` in the triage log

3. **ISSUE_TEMPLATE/agent-issue-body.yml — template field**
   - Added `## Target Repository\n  {target_repo}` section between Proposed Fix Strategy and Action Items

4. **body_templates.py — render_initial_body()**
   - Added `target_repo: str = "_Pending triage_"` parameter
   - Passes it through to `build_body()`

5. **Skill updated** — `pytorch-agent-pipeline` SKILL.md patched with correct xfail/skip pitfall and target_repo visibility note

### Tests: 77/77 passing

### Issues Reverted & Re-verified
- #1951: was DONE (xfail), reverted → DISCOVERED → re-triaged → IMPLEMENTING (torch-xpu-ops)
- #2253: was triage-timeout, re-triaged → IMPLEMENTING (pytorch)  
- #2295: was DONE (skipped), reverted → DISCOVERED → re-triaged → IMPLEMENTING (torch-xpu-ops)
- #2800: re-verified → genuinely FIXED (DONE)
- #2891: re-verified → genuinely FIXED (DONE)

## Current Issue Status (17 issues)

| # | Status | Target Repo | Notes |
|---|--------|-------------|-------|
| 1951 | IMPLEMENTING | torch-xpu-ops | BatchNorm contiguity fix |
| 1963 | IMPLEMENTING | pytorch | autocast registrations |
| 2015 | IMPLEMENTING | torch-xpu-ops | test logic mismatch |
| 2253 | IMPLEMENTING | pytorch | dtypesIfXPU for ~16 OpInfo entries |
| 2295 | IMPLEMENTING | torch-xpu-ops | EmbeddingBag SYCL kernel float64 |
| 2359 | NEEDS_HUMAN | torch-xpu-ops | kernel design work |
| 2436 | NEEDS_HUMAN | pytorch | deep autograd investigation |
| 2512 | IMPLEMENTING | pytorch | SummaryOps.cpp sync |
| 2518 | IMPLEMENTING | torch-xpu-ops | test_torch_xpu.py skip |
| 2554 | NEEDS_HUMAN | pytorch | upstream Triton/Inductor |
| 2609 | IMPLEMENTING | pytorch | cpp_wrapper_cpu.py device shim |
| 2615 | IMPLEMENTING | torch-xpu-ops | FFT promote_fft_input fix |
| 2693 | NEEDS_HUMAN | pytorch | deep Dynamo analysis |
| 2712 | IMPLEMENTING | pytorch | swap_tensors+weakref |
| 2800 | DONE | — | genuinely fixed |
| 2891 | DONE | — | genuinely fixed |
| 2953 | IMPLEMENTING | torch-xpu-ops | XPU fill kernel overflow |

## Remaining Tasks

### Task 1: Backfill Target Repository section (NOT YET DONE)
Issues that were triaged before the template fix don't have the visible `## Target Repository` section.
Need to insert it for: 1963, 2015, 2359, 2436, 2512, 2518, 2554, 2609, 2615, 2693, 2712, 2953

Script approach:
```python
cd ~/torch-xpu-ops/.github/issue-handler
set -a && source .env && set +a
~/pytorch/.venv/bin/python -c "
from issue_handler.utils import git as gh
from issue_handler.utils.body_templates import get_metadata
import re
repo = 'intel/torch-xpu-ops'
for n in [1963, 2015, 2359, 2436, 2512, 2518, 2554, 2609, 2615, 2693, 2712, 2953]:
    detail = gh.get_issue_detail(repo, n)
    body = detail.get('body', '') or ''
    if '## Target Repository' in body:
        print(f'#{n}: already done'); continue
    target = get_metadata(body, 'target_repo')
    if not target:
        print(f'#{n}: no metadata'); continue
    idx = body.find('## Action Items')
    if idx == -1:
        print(f'#{n}: no Action Items'); continue
    new_body = body[:idx] + f'## Target Repository\n{target}\n\n' + body[idx:]
    gh.update_issue_body(repo, n, new_body)
    print(f'#{n}: backfilled = {target}')
"
```

### Task 2: Commit pipeline fixes
```bash
cd ~/torch-xpu-ops
git add .github/issue-handler/issue_handler/verify_existence.py \
       .github/issue-handler/issue_handler/triage_agent.py \
       .github/issue-handler/issue_handler/utils/body_templates.py \
       .github/ISSUE_TEMPLATE/agent-issue-body.yml
git commit -m "fix: verify_existence xfail/skip detection, visible target_repo, run all tests"
```

### Task 3: Run code_fix on 11 IMPLEMENTING issues
11 issues ready for code_fix stage:
- torch-xpu-ops targets (5): 1951, 2015, 2295, 2518, 2615, 2953
  - These use git worktree to avoid disrupting pipeline code
  - Push to origin (intel/torch-xpu-ops), add `disable_all` label
- pytorch targets (5): 1963, 2253, 2512, 2609, 2712
  - Fix in ~/pytorch, push to chuanqi129/pytorch
  - Create draft PR

Run one at a time:
```bash
cd ~/torch-xpu-ops/.github/issue-handler
set -a && source .env && set +a
~/pytorch/.venv/bin/python -m issue_handler.fixing_steps.code_fix --issue N
```

Iterate on one issue first (e.g. #1951), fix bugs, then batch the rest.

### Task 4: Handle special cases
- #2253: has 39 test cases — triage worked this time ($0.09)
- 4 NEEDS_HUMAN issues (2359, 2436, 2554, 2693): skip, they need human intervention

## Environment
- Branch: `agent/pipeline-redesign-v2` in ~/torch-xpu-ops
- Python: `~/pytorch/.venv/bin/python`
- Env: `cd ~/torch-xpu-ops/.github/issue-handler && set -a && source .env && set +a`
- Tests: `~/pytorch/.venv/bin/python -m pytest tests/ -x -q`
