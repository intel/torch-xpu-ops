# Test Plan — Pipeline Redesign v2

## Round 1a: Issue #346 (pytorch, Torch Operations, upstream-pytorch)

Run each stage, fix bugs found, then write tests for bugs encountered.

### Step 1: discovery_agent

```bash
python -m pytorch_agent.discovery_agent --issue 346
```

Verify:
- [x] Issue body reformatted using `agent-issue-body.yml` template
- [x] Sections: Summary, Test Info, Failed Tests, Error Log, Reproducer, Context, Environment, Original Issue
- [x] Original Issue section at bottom (in collapsible `<details>`)
- [x] `<!-- agent:status:DISCOVERED -->` set
- [x] Environment extracted programmatically (no nested `<details>`, no code fences)
- [x] Labels: `agent:discovered` applied

Bugs found (12 total) → 23 tests written. See `tests/test_discovery_agent.py`.

### Step 2: triage_agent

```bash
python -m pytorch_agent.triage_agent --issue 346
```

Verify:
- [x] Root Cause Analysis filled
- [x] Fix Strategy filled
- [x] Verdict → IMPLEMENTING or NEEDS_HUMAN
- [x] `<!-- agent:status:IMPLEMENTING -->` set
- [x] Action Items checkboxes updated

Bugs found:
1. Status gate rejected `DISCOVERED` — only accepted `TRIAGING`/`None`
2. `update_section` regex broken — f-string `#{{{1,4}}}` → `#{(1, 4)}`, sections never matched, always inserted as duplicates

### Step 3: issue_fixing_agent

```bash
python -m pytorch_agent.issue_fixing_agent --issue 346
```

Verify:
- [x] Stage advances correctly (IMPLEMENTING → IN_REVIEW)
- [x] Action Items updated in body (Fix implemented ✅, Fix verified ✅)
- [x] Agent Log entries in body (not comments)
- [x] PR created on `chuanqi129/pytorch` (#6)
- [x] Tracking metadata set (tracking_pr, last_push_sha)
- [x] Correct fix: moved pivot validation from CPU kernel to TORCH_IMPL_FUNC

Bugs found:
1. Missing `parse_sections` import in code_fix.py
2. Duplicate `render_pr_body` import

---

## Round 1b: Issue #342 (torch-xpu-ops, Torch Operations, no dep)

Same steps as Round 1a but targeting torch-xpu-ops repo.

## Round 2: Other Categories (after Round 1 passes)

| Issue | Target Repo | Category | Dependency |
|-------|-------------|----------|------------|
| **#369** | pytorch | Inductor | upstream-pytorch |
| **#311** | torch-xpu-ops | Inductor | triton |
| **#338** | torch-xpu-ops | Torch Operations | oneMKL |
