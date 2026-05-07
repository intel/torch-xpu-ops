# Test Plan — Pipeline Redesign v2

## Round 1a: Issue #346 (pytorch, Torch Operations, upstream-pytorch)

Run each stage, fix bugs found, then write tests for bugs encountered.

### Step 1: discovery_agent

```bash
python -m pytorch_agent.discovery_agent --issue 346
```

Verify:
- [ ] Issue body reformatted using `agent-issue-body.yml` template
- [ ] Sections: Test Info, Error Details, Root Cause Analysis (empty), Fix Strategy (empty)
- [ ] Original Issue section at bottom
- [ ] `<!-- agent:status:DISCOVERED -->` set

Fix any bugs → write tests for those bugs.

### Step 2: triage_agent

```bash
python -m pytorch_agent.triage_agent --issue 346
```

Verify:
- [ ] Root Cause Analysis filled
- [ ] Fix Strategy filled
- [ ] Verdict → IMPLEMENTING or NEEDS_HUMAN
- [ ] `agent:active` label applied

Fix any bugs → write tests for those bugs.

### Step 3: issue_fixing_agent

```bash
python -m pytorch_agent.issue_fixing_agent --issue 346
```

Verify:
- [ ] Stage advances correctly
- [ ] Action Items updated in body
- [ ] Agent Log entries in body (not comments)
- [ ] PR created on `chuanqi129/pytorch`

Fix any bugs → write tests for those bugs.

---

## Round 1b: Issue #342 (torch-xpu-ops, Torch Operations, no dep)

Same steps as Round 1a but targeting torch-xpu-ops repo.

## Round 2: Other Categories (after Round 1 passes)

| Issue | Target Repo | Category | Dependency |
|-------|-------------|----------|------------|
| **#369** | pytorch | Inductor | upstream-pytorch |
| **#311** | torch-xpu-ops | Inductor | triton |
| **#338** | torch-xpu-ops | Torch Operations | oneMKL |
