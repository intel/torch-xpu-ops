# Plan: Redesign Agent Pipeline — Discovery → Triage → Fix → PR

Date: 2026-05-06
Status: DRAFT v3 — issue-body-driven state

## Overview

Redesign the fixing agent into 4 distinct agents, each with its own skill. Issues live in `ZhaoqiongZ/torch-xpu-ops-exp` (accessed via `REVIEW_GH_TOKEN`). Many issues have labels but sparse/unformatted bodies.

**Key design change:** All state lives in the issue body (not comments). Each agent reads and edits the issue body directly — status, action items, logs, root cause, etc. are all sections of the body. No `state.py` comment-based tracking.

```
Raw Issue (sparse body + labels)
        ↓
[Discovery Agent + skill]  →  Formatted issue body (overwritten in-place)
        ↓
[Triage Agent + skill]    →  Root cause added to issue body, stage updated
        ↓
[Fix Agent + skill]       →  Code changes on agent/issue-N branch
        ↓
[PR Pipeline]             →  private review → public PR → CI watch → close
                              (existing logic, mostly unchanged)
```

## Source repo & access

- **Issues repo:** `ZhaoqiongZ/torch-xpu-ops-exp`
- **Token:** `REVIEW_GH_TOKEN` (from .env)
- **Labels already on issues:** `agent_test: ut` / `agent_test: e2e`, `agent_category: *`, `agent_dependency: *`, `skipped`

## Issue body template (after discovery agent formats it)

Based on `.github/ISSUE_TEMPLATE/ci-failure-tracking.yml` plus additions:

```markdown
## Status
<!-- agent:status -->
Stage: DISCOVERED | TRIAGING | IMPLEMENTING | IN_REVIEW | ...
Last updated: <timestamp>
<!-- /agent:status -->

## Summary
[one-line description of the failure]

## Test Type
UT / E2E

## Category
[from label: Torch Operations / Inductor / Distributed / TorchAO / ...]

## Dependency
[from label: upstream-pytorch / oneDNN / triton / ...]

## Platform
PVC / ATS-M / DG2 / BMG / ...

## Failed Tests
- `test/path/to/test.py::TestClass::test_method`

## Error Log
```
[relevant error output, ~50 lines]
```

## Reproducer
```bash
[commands to reproduce locally]
```

## Commit Scope
[if available: last pass → first fail, compare link]

## Root Cause Analysis
<!-- filled by triage agent -->
[analysis of why the failure happens, which component is responsible]

## Proposed Fix Strategy
<!-- filled by triage agent -->
[high-level approach: tolerance fix / kernel fix / skip removal / etc.]

## Action Items
- [ ] 🔍 Issue formatted (Discovery Agent)
  <details><summary>Discovery log</summary>
  <!-- agent:discovery-log -->
  </details>
- [ ] 🧠 Root cause identified (Triage Agent)
  <details><summary>Triage log</summary>
  <!-- agent:triage-log -->
  </details>
- [ ] 🔧 Fix implemented (Fix Agent)
  <details><summary>Fix log</summary>
  <!-- agent:fix-log -->
  </details>
- [ ] ✅ Fix verified locally (Fix Agent)
- [ ] 📋 PR proposed (PR Pipeline)
- [ ] 👀 Human review
- [ ] 🎉 PR merged
```

<!-- REVIEW template: ✅ Updated — all state in issue body, no comment-based tracking -->

## Agent 1: Discovery Agent

**Input:** Raw issue with labels, possibly sparse body
**Input:** Raw issue with labels, possibly sparse body
**Output:** Issue body overwritten with formatted template above
**Skill:** `pytorch-issue-discovery`
What it does:
1. Read the issue body + labels
2. If body is already formatted (has `<!-- agent:status -->` marker), skip
3. Extract whatever info exists: test names, error logs, reproducer, context
4. If info is missing, infer from labels + test paths (e.g., test path → reproducer command)
5. Overwrite issue body with the formatted template (status = `TRIAGING`, "Issue formatted" checked)
6. Add label `agent:active` to the issue

The skill teaches the LLM:
- How to parse raw issue formats (some have `### 🐛 Describe the bug`, some have structured sections, some are just test lists)
- How to construct reproducer commands from test paths
- What each label category means
- The output template format

<!-- REVIEW Agent 1: ✅ Updated — adds agent:active label -->

## Agent 2: Triage Agent

**Input:** Formatted issue
**Output:** Root cause analysis + fix strategy added to issue body
**Skills:** `pytorch-triage-ut` (for UT), `pytorch-triage-e2e` (for E2E)

What it does:
1. Read the formatted issue
2. Analyze the error log, test code, and relevant pytorch source
3. Update issue body: fill Root Cause Analysis section, fill Proposed Fix Strategy section, check "Root cause identified", update status to `IMPLEMENTING` or `NEEDS_HUMAN`, append triage log to folded details
4. Update labels: keep `agent:active` or swap to `agent:needs-human`

Two skills because UT and E2E need different approaches:
- **UT triage:** look at test source, trace the failing op/kernel, check XPU backend implementation
- **E2E triage:** look at model config, check if it's a compilation issue, accuracy issue, or missing op

<!-- REVIEW Agent 2: ✅ Updated — agent:active / agent:needs-human labels -->

## Agent 3: Fix Agent

**Input:** Formatted issue with root cause + fix strategy
**Output:** Code changes on `agent/issue-N` branch
**Skill:** `pytorch-fix`

What it does:
1. Read the issue (now has summary, root cause, fix strategy, reproducer)
2. The LLM has all context it needs — no more prompt building in Python
3. Implement the fix in `~/pytorch` or `~/torch-xpu-ops` (depending on the fix)
4. Run the reproducer to verify
5. Commit and push to review remote
6. Update issue body: check "Fix implemented" + "Fix verified locally", update status to `IN_REVIEW`, append fix log to folded details
7. Keep label `agent:active`

The skill teaches:
- pytorch code conventions
- Common XPU fix patterns (tolerance adjustment, kernel registration, op fallback)
- What NOT to do (no `@skipIfXpu`, no third_party changes, no force-push)

<!-- REVIEW Agent 3: ✅ Updated — keeps agent:active label -->

## Agent 4: PR Pipeline (existing, minimal changes)

The existing `private_review.py → public_submit.py → ci_watch.py → close_issue.py` flow.

Changes needed:
- Make source issue repo configurable in `config.py` (env var `ISSUE_REPO`, default `ZhaoqiongZ/torch-xpu-ops-exp`)
- Make token selection automatic based on repo (already have `_token_for_repo()`)
- Update action items checklist as stages complete

<!-- REVIEW Agent 4: ✅ Updated — repo is configurable via ISSUE_REPO env var -->

## File structure changes

```
pytorch_agent/
  discovery_agent.py        # NEW — Agent 1
  triage_agent.py           # RENAMED from issue_triaging_agent.py, rewritten
  issue_fixing_agent.py     # Orchestrator — dispatches to discovery/triage/fix/PR steps
  fixing_steps/
    implement.py            # Slim wrapper: setup branch → call fix agent → push
    private_review.py       # Mostly unchanged
    public_submit.py        # Mostly unchanged
    ci_watch.py             # Mostly unchanged
    close_issue.py          # Mostly unchanged
  utils/
    issue_body.py            # NEW — read/write/update sections of the issue body
    config.py                # Updated — ISSUE_REPO configurable
    github_client.py         # Updated — update_issue_body() public method
    state.py                 # DEPRECATED — replaced by issue_body.py
    ...                      # Other utils unchanged

skills/
  pytorch-issue-discovery/SKILL.md    # NEW
  pytorch-triage-ut/SKILL.md          # NEW
  pytorch-triage-e2e/SKILL.md         # NEW
  pytorch-fix/SKILL.md                # NEW (or update existing)
```

<!-- REVIEW File structure: ✅ Updated — no separate fix_agent.py, implement.py is the slim wrapper that calls the LLM with the fix skill -->

## Proof of concept scope

Pick ~2 issues per category for initial testing:

| Category | Example issues | Test type |
|----------|---------------|-----------|
| Torch Operations | #378 (index_add accuracy), #370 | UT |
| Inductor | #375 (streams), #374 | UT |
| TorchAO | #379 (quantized model), #371 | UT |
| Torch Runtime | #377 (multiprocessing), #359 | UT |
| E2E | #365 (int8 perf), #364 (int8 accuracy) | E2E |

<!-- REVIEW PoC scope: -->

## Open questions

1. Should `discovery_agent.py` and `triage_agent.py` be standalone CLI scripts (like current `issue_triaging_agent.py`) AND callable from the orchestrator?
2. For the PoC, should we run all 4 agents end-to-end on one issue first, or build/test each agent independently?
3. The skills — should they live in `torch-xpu-ops/.github/skills/` (repo skills) or `~/.hermes/skills/` (hermes skills)? Or in the pytorch-agent directory itself?

<!-- REVIEW Open questions: -->
