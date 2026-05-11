# E2E Test Plan — Agent Pipeline

## Goal

Validate the fully autonomous agent pipeline end-to-end: **cron discovers issues → format_agent → triage_agent → fix_agent → PR submitted → human review → "/agent LGTM"**.

Hermes acts as **monitor only** — it observes the pipeline, diagnoses failures, and fixes infrastructure/skill/config problems. Hermes **never** solves the issues themselves.

---

## Architecture Under Test

```
┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│  Cron   │────▶│ format_agent │────▶│ triage_agent │────▶│ fix_agent │
│ (fetch) │     │ (DISCOVERED) │     │  (TRIAGED)   │     │ (FIXING)  │
└─────────┘     └──────────────┘     └──────────────┘     └───────────┘
                                                                │
                                                                ▼
                                                          ┌───────────┐
                                                          │  PR Agent │
                                                          │(PR_READY) │
                                                          └─────┬─────┘
                                                                │
                                                                ▼
                                                          ┌───────────┐
                                                          │  Review   │
                                                          │"/agent    │
                                                          │  LGTM"   │
                                                          └───────────┘
```

## Success Criteria

An issue is **PASSED** when:
1. Cron picks it up automatically (no manual trigger)
2. format_agent formats the issue body (status → DISCOVERED)
3. triage_agent assigns correct labels and writes root cause analysis (status → TRIAGED)
4. fix_agent creates a branch with a fix (status → FIXING)
5. PR is submitted to `ZhaoqiongZ/torch-xpu-ops-exp` or `chuanqi129/pytorch`. Depending on the issue.
6. Reviewer comments `/agent LGTM` on the PR

An issue is **FAILED** when:
- Any agent step hangs, crashes, or produces no output
- Labels are clearly wrong (e.g. Inductor issue labeled as Distributed)
- Fix is nonsensical or doesn't compile
- Pipeline gets stuck and doesn't advance to the next stage
- The agent loop is exceeded.

## Hermes Monitor Role

### DO
- Watch cron logs, agent logs, and issue status transitions
- Diagnose and fix: skill files, `.opencodeignore`, config, backend bugs, timeout issues
- Patch skills when agent output quality is poor due to bad instructions
- Fix cron scheduling if jobs don't fire
- Report status summaries to user
- Watch the labels on each step is tagged correctly.

### DO NOT
- Run agents manually (no `python -m issue_handler.format_agent --issue X`)
- Write code fixes for the issues
- Push branches or create PRs on behalf of agents
- Modify issue bodies directly

---

## Starter Scope — 8 Issues

Covers 7/8 top-level categories, 6/8 dependencies, both test types.

| # | Issue | Title (short) | Expected Labels |
|---|-------|---------------|-----------------|
| 1 | #191 | logspace accuracy failures | `Torch Operations` · `eltwise` · `ut` |
| 2 | #1662 | dynamo test_misc_xpu sub_issue | `Inductor` · `dynamo` · `upstream-pytorch` · `ut` |
| 3 | #327 | distributed accuracy error xccl | `Distributed` · `xccl` · `ut` |
| 4 | #258 | INT4 Flex-attention perf drop | `TorchAO` · `triton` · `ut` |
| 5 | #273 | work_group_scratch RuntimeError | `Flash Attention` · `oneAPI` · `ut` |
| 6 | #278 | hspmm NotImplementedError | `Sparse` · `CPU fallback` · `ut` |
| 7 | #361 | FX profiler tests fail on XPU | `Torch Runtime` · `upstream-pytorch` · `ut` |
| 8 | #365 | PT2E INT8 perf regression | `Inductor` · `lowering` · `oneDNN` · `e2e` |

### Coverage Matrix (Starter)

| Dimension | Covered | Missing |
|-----------|---------|---------|
| Top-level category (8) | 7 | Others |
| Inductor phase (5 active) | 2: dynamo, lowering | codegen, runtime, test_infra |
| Torch ops sub (6 active) | 1: eltwise | reduction, gemm, conv, multi_ops, other |
| Dependency (8) | 6 | oneMKL, driver |
| Test type (2) | 2: ut × 7, e2e × 1 | — |

### Execution Steps

1. **Pre-flight checks**
   - [ ] `.opencodeignore` exists in `~/pytorch`
   - [ ] `.env` has valid tokens (`REVIEW_GH_TOKEN`, `GH_TOKEN`)
   - [ ] 45/45 unit tests pass
   - [ ] Dev branch (`agent/pipeline-redesign-v2`) is up to date
   - [ ] No stale opencode processes running

2. **Cron setup**
   - [ ] Configure cron to scan these 8 issues
   - [ ] Verify cron fires and picks up issues
   - [ ] Verify `agent:active` label is applied during processing

3. **Monitor loop** (Hermes)
   - Poll issue status every 5 min
   - Check agent logs for hangs (no output > 5 min)
   - Check for `agent:blocked` or `agent:needs-human` labels
   - Check for the agent loop times. If any issue exceeds 3 loops, This is marked as failed. Hermes should record the reason for failure.
   - Report progress table to user

4. **Review phase**
   - Once PRs appear, user reviews
   - `/agent LGTM` = passed
   - Review comments trigger fix_agent to amend (one commit per comment)

5. **Post-mortem**
   - Collect pass/fail per issue
   - Document any skills patched, config changed, bugs found
   - Decide whether to proceed to full scope

---

## Full Scope — 35 Issues

Expand after starter scope passes ≥ 6/8 issues.

### Top-Level Categories (2 per category = 16 issues)

| Category | Issue 1 | Issue 2 |
|----------|---------|---------|
| Torch Operations | #378 (index_add_ accuracy) | #368 (vmap addmv, +oneDNN) |
| Inductor | #1662 (dynamo test_misc) ★ | #373 (AOTAutogradCache, +codegen) |
| Distributed | #327 (accuracy error) ★ | #306 (_reset_fr_recording) |
| TorchAO | #379 (test_ts2ep, +upstream) | #258 (INT4 Flex-attention) ★ |
| Flash Attention | #334 (sdpa export crash) | #333 (sdpa crash, +oneDNN) |
| Sparse | #278 (hspmm NotImpl) ★ | #281 (sampled_addmm, +triton) |
| Torch Runtime | #377 (multiprocessing) | #361 (FX profiler) ★ |
| Others | #355 (torchbind, +upstream) | #351 (CI false failures) |

★ = already in starter scope

### Inductor Phase (1 per phase = 5 issues)

| Phase | Issue |
|-------|-------|
| dynamo | #1662 ★ |
| inductor_lowering | #365 ★ |
| inductor_codegen | #345 (wrong grad torch.compile) |
| runtime | #177 (Cannot swap t2 weak ref) |
| test_infra | #307 (test_ctx_manager regression) |

### Torch Ops Sub-type (1 per sub = 6 issues)

| Sub-type | Issue |
|----------|-------|
| eltwise | #191 ★ |
| reduction | #331 (atomic CAS clarification) |
| gemm | #197 (broadcast shape mismatch) |
| conv | #201 (depthwise_conv 64bit indexing) |
| multi_ops | #342 (empty source out-of-bounds) |
| other | #370 (pool3d large int64) |

### Agent Dependency (1 per dep = 8 issues)

| Dependency | Issue |
|------------|-------|
| upstream-pytorch | #355 (torchbind) |
| oneDNN | #350 (GPT2_large fp16 inference) |
| oneMKL | #303 (stft accuracy gap) |
| oneAPI | #273 ★ |
| triton | #311 (slice_scatter wrong grad) |
| driver | #328 (ocloc/IGC compilation) |
| xccl | #327 ★ |
| CPU fallback | #366 (new failed test cases) |

### Agent Test Type (1 per type = 2 issues)

| Test Type | Issue |
|-----------|-------|
| ut | #191 ★ |
| e2e | #348 (BMG dynamo benchmark regression) |

### Full Scope — Unique Issues

After dedup (★ = shared with starter): **~28 new + 8 starter = ~35 total unique issues**

### Full Scope Coverage

| Dimension | Covered | Missing |
|-----------|---------|---------|
| Top-level category (8) | 8/8 | — |
| Inductor phase (5 active) | 5/5 | — |
| Torch ops sub (6 active) | 6/6 | — |
| Dependency (8) | 8/8 | — |
| Test type (2) | 2/2 | — |

**100% coverage of all non-empty agent labels.**

---

## Skipped Labels (0 issues exist)

These labels have no open issues assigned. Skip for now, revisit when issues appear:

- `agent_category_inductor_phase: fx_graph`
- `agent_category_inductor_phase: unknown`
- `agent_category_torch_ops_not_op_impl`

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| opencode hangs on ~/pytorch | `.opencodeignore` auto-provisioned; select()-based timeout kills stuck processes |
| Triage agent too slow (>10 min) | Increased timeout to 900s; monitor log growth |
| Fix agent produces bad code | Review gate — human must `/agent LGTM` before merge |
| Cron misses issues | Monitor cron logs; verify label transitions |
| Token rate limits | Space cron intervals; max 2 concurrent agents |
| Upstream-dependent issues can't be fixed locally | Agent should label `agent:blocked` + `agent_dependency: upstream-pytorch` and skip fix step |

---

## Metrics to Track

| Metric | Definition |
|--------|------------|
| **Format success rate** | Issues that reach DISCOVERED / total |
| **Triage accuracy** | Labels match expected / total labels |
| **Fix success rate** | PRs that compile and address the issue / total TRIAGED |
| **End-to-end time** | Time from cron pickup to `/agent LGTM` |
| **Hermes interventions** | Count of infrastructure fixes needed |
| **Skill patches** | Count of skill updates triggered by test findings |
