# classify_ut skipped/xfail cases

This skill follows agent-guidelines AND extends it with skipped-XPU-status UT classification rules.

## Purpose

Classify rows whose `Reason` is blank and `status_xpu` is `skipped` or `xfail`.
Skipped does **not** automatically mean a test environment limitation. Every skipped row requires
semantic analysis of `message_xpu`, linked issues, local source, and targeted local runs.

This subskill is also invoked for the **TBE re-verification pass**: rows where
`TBE_Reverify = True` in the `.agent.xlsx` (originally classified as `To be enabled`,
opt-in recheck) get routed here by `status_xpu`. The deep-analysis workflow is the same;
the only difference is the input row set and the `DetailReason` marker convention. See
**TBE Re-verification Context** below and the **TBE Re-verification Rule** in
`classify_ut/RULES.md`.

## TBE Re-verification Context

When the parent workflow runs the re-verification pass, it routes a row with
`TBE_Reverify = True` and `status_xpu` in {`skipped`, `xfail`} to THIS subskill. The
invocation is identical to the skipped/xfail-Reason case with one exception: the row is
being re-checked, not classified from scratch.

- The existing `DetailReason` was written by a prior pass that landed on
  `To be enabled`. The agent MUST re-read the cited source state — typically a stale
  `skipIfXpu` decorator, a closed `skipped`-labeled issue whose decorator is still in
  place, a missing `allow_xpu=True`, or a stale `inductor_skips["xpu"][...]` entry —
  and check whether it has changed since the prior verdict.
- For `skipped`-label citations specifically: re-run `gh issue view` on the cited
  closed issue to confirm it is still CLOSED. If it has been reopened, the prior TBE
  rationale no longer holds and the verdict may change.
- The verdict may stay `To be enabled`, change to another canonical label (e.g. the
  decorator was removed and the case now runs but crashes -> `Failures (xpu broken)`),
  or be flagged `Need human check` (LOW) when the cited signal is no longer clear. Each
  outcome is written to `Reason` and `DetailReason` per the **TBE Re-verification Rule**.
- `Reason TBD` is **never** flipped to `True` for a re-verified row. A re-verified row
  stays `Reason TBD = False`.
- The updated `DetailReason` MUST start with `[Reverified: YYYY-MM-DD]` so a human
  reviewer can distinguish a row that was re-checked in this session from one that was
  left as-is from the original workbook. If the row also flips to `Need human check`,
  the LOW confidence prefix is required by the existing rubric:
  `[Reverified: YYYY-MM-DD] [Confidence: LOW] Need human check. ...`
- The `Confidence` workbook column is NOT populated for re-verified rows (it is reserved
  for `Reason TBD = True` rows; see the **Confidence Rubric & Need-Human-Check Rule** in
  `RULES.md`).

## Required Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `read` | Inspect test methods, decorators, skip helpers | Read `test_torchinductor.py` line 4110 |
| `bash` | Run tests locally, activate env | `python test_file.py -k pattern -v` |
| `grep` | Find decorators, skip entries, `TestFailure` dicts | `grep -n 'skipIfXpu' test_file.py` |
| `gh` CLI | Fetch issue state, search issues | `gh issue view 2334 --repo=intel/torch-xpu-ops --json=state,closedAt` |
| `gh search` | Find known issues by keyword | `gh search issues "test_div7 xpu" --repo=intel/torch-xpu-ops` |

## Hard Constraints

- Do not treat `status_xpu = skipped` as environment issue by default.
- Do not classify by message pattern alone. The message points to evidence; it is not the conclusion.
- Always check linked issue state (OPEN vs CLOSED) before classifying.
- If `message_xpu` is empty or nondiagnostic, run the test locally.
- Local runs use `pytorch_opencode_env` and the configured source checkout (`PYTORCH_SRC`,
  default `$HOME/upstream/pytorch`). For source-aligned results, optionally run the
  preparation skill first (`classify_ut/preparation/SKILL.md`, Environment Setup) — not
  required by default.
- Do not change `Reason TBD` after classification.
- Mark updated cells blue.
- For every row with `Reason TBD = True`, `DetailReason` MUST start with a confidence prefix
  `[Confidence: HIGH|MEDIUM|LOW]` per the **Confidence Rubric & Need-Human-Check Rule** in the
  parent skill. When the rubric resolves to LOW, set `Reason = "Need human check"` and
  enumerate which axes were checked (sheet match, base function, XPU wrapper, known issue,
  local run) and why each was inconclusive.
- When a candidate test maps to an `Issues`-sheet row carrying the `skipped` label (and not
  also `not_target` / `wontfix`), follow the **Dynamic-Skip Rule** in the parent skill:
  local verification via `port-pytorch-tests-xpu` on a `daisyden/pytorch` branch is REQUIRED
  to split `Failures (xpu broken)` vs `Feature gap` vs (rarely) `To be enabled`.

## User-Issued Policy Overrides (mirror of parent skill — AUTHORITATIVE)

Apply these on every skipped/xfail row. P2 and P3 below are AUTHORITATIVE and live here (they
were moved out of the parent skill). The parent rules file `classify_ut/RULES.md` carries the
full text for P1 and the **Sibling-Class Verdict Mapping**.

- **P1 Local-Retest Mandate:** Local retest is REQUIRED whenever the row has no linked issue,
  no linked PR, and no `not_target`/`wontfix` evidence — even when `message_xpu` or a source
  line alone would suggest a verdict. Source citations have repeatedly disagreed with
  on-device behaviour (dynamic skips, dtype-only gates).
- **P2 — `Not applicable` Must Carry Evidence, But Must NOT Be Run (AUTHORITATIVE):**
  - Do not execute `Not applicable` rows that already carry valid evidence (linked
    `not_target`/`wontfix` issue, owner-team scope, or a clearly CUDA-only API such as
    `torch.cuda.default_generators[0]`).
  - EVERY `Not applicable` classification MUST cite explicit evidence in `DetailReason`. A
    classification with no evidence is invalid — re-route to `To be enabled` or run locally.
- **P3 — JIT Cluster Policy (AUTHORITATIVE):** JIT (`oncall:jit`, `test_jit*`, `torch.jit.*`)
  clusters are `Not applicable` by owner-team scope. Cite the owner-team label as evidence in
  `DetailReason`. Do NOT classify JIT clusters as `Local Passed`, even when the test passes on
  XPU.
- **Sibling-Class Verdict Mapping (`instantiate_device_type_tests`):** XPU is NOT in the
  default test-base set. The XPU sibling is generated ONLY when the call site passes
  `allow_xpu=True`. Therefore: (1) if the source file uses `instantiate_device_type_tests`
  WITHOUT `allow_xpu=True`, the verdict is `To be enabled` — cite the file:line of the call
  and state that `allow_xpu=True` must be added before any XPU verdict is reachable; (2) if
  `allow_xpu=True` IS present, run the XPU sibling locally and map outcomes: PASS ⇒
  `Local Passed`, FAIL/SIGSEGV ⇒ `Failures (xpu broken)`, `skipIfXpu` ⇒ Dynamic-Skip Rule.
  `Not applicable` is NEVER the default for sibling-pattern cases — it requires its own
  evidence (linked `not_target`/`wontfix` issue, owner-team scope with CUDA-only dispatch, or
  CUDA-specific API in the test body). Workbook classname may differ from source classname;
  alias before lookup. See the parent skill **Sibling-Class Verdict Mapping** section for the
  full decision flow.
- **Workbook Precedent Rule (parent skill):** A non-Inductor TBD row may adopt a peer cluster's
  verdict when at least 5 same-`testfile_cuda` peers are ≥95% unanimous on one canonical Reason.
  This is a MEDIUM-confidence override that still requires at least one independent source-evidence
  axis in `DetailReason`. See the **Workbook Precedent Rule** section in `classify_ut/RULES.md`.

## Optional Preparation Inputs (Environment Setup + Local Pre-Screen)

The parent-skill preparation steps (`classify_ut/preparation/SKILL.md`) are **OPTIONAL and
NOT run by default**. For skipped/xfail rows the authoritative evidence is the
closed-issue rule below; pre-screen results are additive, not required. This subskill works
whether or not preparation was run:

- **If Environment Setup was run**: a provenance record at
  `~/.claude_classify_ut_session_provenance.json` records the aligned source SHAs and
  `xpu_available`. Prefer those aligned checkouts for source analysis.
- **If Local Pre-Screen was run**: a `local_result` column exists adjacent to `DetailReason`.
  - Rows whose `local_result` starts with `PASS;` are **TERMINALLY classified** as
    `Reason = "Local Passed"`. This subskill MUST NOT touch them.
  - For `FAIL`, `ERROR`, `TIMEOUT`, `SKIP`, or `SEGFAULT`, the value and its log path are
    additive evidence for the eventual `DetailReason`.
- **If preparation was NOT run**: proceed directly with the closed-issue rule and
  case-existence analysis in this subskill.

See the parent skill **Optional Preparation** section and
`classify_ut/preparation/SKILL.md` for the full helper-script invocation, log path
convention, and 60-second per-test timeout policy.

## Closed-Issue Rule (AUTHORITATIVE for skipped rows)

**When a skipped/xfail row references an issue in `pytorch/pytorch` OR `intel/torch-xpu-ops`
(including `intel/intel-xpu-backend-for-triton`) and that issue is CLOSED, the verdict is
unconditionally `Reason = "To be enabled"`.**

This rule applies regardless of:
- which decorator/mechanism produced the skip (`skipIfXpu`, `@unittest.skip`, PyTorch
  disabled-test, `inductor_skips`, `TestFailure` dict, dynamic `raise SkipTest`, etc.),
- whether the issue lives in `pytorch/pytorch`, `intel/torch-xpu-ops`, or
  `intel/intel-xpu-backend-for-triton`,
- whether the closing PR has actually removed the skip decorator yet,
- whether a local pre-screen result is available (the closed-issue evidence is sufficient
  on its own; pre-screen results are additive, not required).

### Carve-outs (do NOT override)

The rule does NOT override:
- `Reason = "Local Passed"` — `Local Passed` is a terminal verdict backed by actual local
  execution and is stronger evidence than a closed issue. Leave these rows untouched.
- `Reason = "Not applicable"` whose `DetailReason` cites a `not_target` / `wontfix` label on
  the `Issues` sheet — out-of-scope decisions outrank the closed-issue rule.

If a row's only referenced issues are ALL closed AND the current `Reason` is one of
`Failures (xpu broken)`, `Feature gap`, `Community Change`, or `Test Enviroment limitation`,
the rule applies and the row MUST be re-classified as `To be enabled`.

Rationale: a closed pytorch/torch-xpu-ops/triton issue means the upstream fix landed; the
remaining skip is stale tooling debt that should be reaped by re-enabling the case. The
correct routing is `To be enabled`, never `Failures (xpu broken)`, `Feature gap`,
`Test Enviroment limitation`, or `Not applicable`.

### `DetailReason` requirements

`DetailReason` MUST cite:
1. The full issue URL (never a bare `#NNNN`).
2. The closed state and `closedAt` date verified via `gh issue view <N> --repo=<owner>/<repo>
   --json=state,closedAt`.
3. The skip mechanism that needs to be removed (decorator + file:line, or skip-dict key).

Example:
```
[Confidence: HIGH] To be enabled. https://github.com/intel/torch-xpu-ops/issues/2334
CLOSED 2026-03-25; @skipIfXpu at test/inductor/test_torchinductor.py#L4110 not yet
removed.
```

### Verification command

For every issue URL extracted from `message_xpu`, source decorators, or skip-dict comments:

```bash
gh issue view <N> --repo=<owner>/<repo> --json=state,closedAt,title
```

- `state == "CLOSED"` → apply this rule → `Reason = "To be enabled"`.
- `state == "OPEN"` → fall through to Step 4 decision tree (Feature gap / Failures /
  local-verify branch).

### Anti-patterns

- Treating a CLOSED pytorch/torch-xpu-ops/triton issue as `Failures (xpu broken)` because
  the decorator is still present. The decorator is stale; the verdict is `To be enabled`.
- Demanding a local-run artifact to confirm a CLOSED-issue verdict. The closed state IS
  the HIGH evidence; local runs are optional corroboration.
- Citing the issue number without the full URL or without the verified `closedAt` date.
- Applying this rule to issues in unrelated repos (e.g., `triton-lang/triton` upstream,
  CCCL, oneDNN). Only `pytorch/pytorch`, `intel/torch-xpu-ops`, and
  `intel/intel-xpu-backend-for-triton` are in scope.

## Workflow Steps

### Step 1: Confirm Eligibility
- `Reason` is blank
- `status_xpu` is `skipped` or `xfail`
- CUDA/XPU metadata identifies the exact test case

### Step 1.5: Check for Community Change Regression

If `last_status_xpu = passed` (test previously passed on XPU but is now skipped):

1. This is likely a regression from an upstream commit. Check git log first:
   ```bash
   cd "$PYTORCH_SRC"
   git log --oneline -20 -- <testfile_cuda>
   ```
2. For candidate commits, inspect the diff:
   ```bash
   git show <commit_hash> -- <testfile_cuda>
   ```
3. Look for changes that would cause the skip: new skip decorator, renamed method,
   changed parametrization, restructured test class, new `TestFailure` entry, etc.
4. If a guilty commit is found:
   - Reason: `Community Change`
   - DetailReason: `Community commit <short_hash> (<author>, <date>) - <summary>.`
     Include git log + git show evidence, how the commit caused the skip
5. If no relevant commit found, continue to Step 2 (normal skip analysis).

### Step 2: Check for CPU Test (Priority Rule)

Before analyzing the skip message, check if this is a CPU test:
- Test name ends with `_cpu` or contains `_cpu_` (e.g., `test_fp8_cpu`, `test_while_loop_with_parameters_cpu`)
- Test has `cpu` as device parameter in parametrization
- Skip message says "requires GPU", "requires a GPU", "GPU_TYPE", or similar

If ANY of the above is true:
- Reason: `Not applicable`
- DetailReason: `CPU Case. CPU test (device=cpu per test name/parametrization), not relevant to XPU validation`
- Skip remaining steps.

### Step 3: Analyze `message_xpu`

Parse the skip message to identify the skip mechanism. **When `message_xpu` contains a URL,
always extract and include the full URL in `DetailReason`.**

| Message Pattern | Skip Mechanism | Next Action |
|----------------|----------------|-------------|
| `Test is disabled because an issue exists disabling it: <URL>` | PyTorch disabled-test | Extract URL -> `Community Change`, DetailReason = full URL |
| `skipIfXpu: <reason>, <issue_url>` | `@skipIfXpu` decorator | Extract URL -> Check issue state |
| `Test is disabled because an issue exists: <url>` | PyTorch disabled-test | Extract URL -> `Community Change`, DetailReason = full URL |
| `test is slow; run with PYTORCH_TEST_WITH_SLOW` | Slow test gate | Run with `PYTORCH_TEST_WITH_SLOW=1` |
| `Requires at least N GPUs` | Hardware requirement | `Test Enviroment limitation` |
| `not-support-multithread` | XPU feature gap | `Feature gap` + #3098 |
| `Only runs on cuda` | CUDA-only gate | Apply the **CUDA-Only Judgement Rule** (read that section in `classify_ut/RULES.md`): look up the API in the `Operation/API` column of the `Not applicable` sheet of `torch_xpu_ops_issues.xlsx` via `classify_ut/scripts/list_not_applicable.py`. Match -> `Not applicable` with `Issue ID` and `Category` cited. No match -> `To be enabled` (often a stale gate or SM-capability check). |
| `Skipped!` (generic) | Various mechanisms | Read source to find skip dict/decorator |
| `Fails with Triton update` | Unconditional `unittest.skip` | `Test Enviroment limitation` (all backends) |
| `Fails under GCC 13` | Compiler version | `Test Enviroment limitation` |
| `sm89 errors out` / `SM90OrLater` | CUDA compute capability gate | `To be enabled` (not CUDA-specific) |
| Empty / `Skipped test` / `xfail` | Unknown | Run locally + read source |

### Step 4: Check Issue State

For ANY issue URL found in the message or source:

```bash
gh issue view <number> --repo=<org/repo> --json=state,title,closedAt
```

**Decision tree based on issue state:**

```
Issue CLOSED?
  YES -> Apply Closed-Issue Rule (above): Reason = "To be enabled"
         unconditionally. Local run is optional, not required.
  NO (OPEN) ->
    Issue describes missing feature? -> "Feature gap"
    Issue describes broken implementation? -> "Failures (xpu broken)"
    Issue describes flaky CI? -> Run locally
      PASSES -> "Local Passed"
      FAILS  -> "Failures (xpu broken)"
```

### Step 5: Handle `Skipped!` Without Clear Message

When `message_xpu` is just `Skipped!` with no explanation:

1. **Find the skip source** in the test file:
   ```bash
   grep -n 'TestFailure\|inductor_skips\|skipIfXpu\|unittest.skip' test/inductor/<file>.py
   ```

2. **Read the skip mechanism**:
   - `test_failures_*` dict with `is_skip=True` -> in-tree XPU skip
   - `inductor_skips["xpu"]` -> inductor dtype/op skip
   - `unittest.skip("reason")` -> unconditional skip (all backends)
   - Dtype restriction in test body (`raise unittest.SkipTest("Skipped!")`) -> check allowed dtypes

3. **Search for known issues**:
   ```bash
   gh search issues "<test_name> xpu" --repo=intel/torch-xpu-ops --limit=5
   gh search issues "<test_name> xpu" --repo=pytorch/pytorch --limit=5
   ```

4. **Try running without the skip**:
   - For `inductor_skips`: test the op directly via `torch.compile`
   - For `TestFailure` dicts: run the base test in the parent test file
   - For dtype restrictions: check if the dtype actually works on XPU

5. **Classify based on results**:
   - Base test passes -> `To be enabled` (skip is stale)
   - Base test fails with known issue -> `Failures (xpu broken)` + issue link in DetailReason
   - Base test fails without known issue after searching both repos -> `Failures (xpu broken)` +
     `[Issue_TBD]` prefix in DetailReason

### Step 6: Handle Slow Tests

```bash
source "${CONDA_ACTIVATE:-$HOME/miniforge3/bin/activate}" "${PYTORCH_ENV:-pytorch_opencode_env}" && \
PYTORCH_TEST_WITH_SLOW=1 python test/inductor/<file>.py -k "<test_name>" -v
```

- PASSES -> `Local Passed`, detail: `Local verification passed with PYTORCH_TEST_WITH_SLOW=1`
- FAILS -> Search known issues, classify as failure
- 0 tests collected -> `Not applicable` (test removed)

### Step 7: Handle PyTorch Disabled-Test Issues

When `message_xpu` says `Test is disabled because an issue exists disabling it: <URL>`:
- The test was disabled by the PyTorch CI infrastructure via `.pytorch-disabled-tests.json`.
- This IS a community change -- the community disabled the test; no local run is required.
- Extract the full issue URL from `message_xpu` and cite it in `DetailReason`.
- Reason: `Community Change`
- DetailReason: `<full URL>. Test previously passed (last_status_xpu=passed) but disabled by the
  PyTorch disabled-test mechanism (<platform(s) from message>).`

**Example:**
```
message_xpu: "Test is disabled because an issue exists disabling it: https://github.com/pytorch/pytorch/issues/180324 for platform(s) linux, slow"
Reason: Community Change
DetailReason: https://github.com/pytorch/pytorch/issues/180324. Test previously passed (last_status_xpu=passed) but disabled by PyTorch disabled-test mechanism (platform: linux, slow)
```

### Step 8: Handle `skipIfXpu` with Closed Issues

When `skipIfXpu` references a CLOSED issue:
- The issue is FIXED but the `skipIfXpu` decorator hasn't been removed yet
- Classify as `To be enabled`
- DetailReason: `<issue_url> (CLOSED <date>) - skipIfXpu decorator not yet removed; issue is fixed`

### Step 9: Handle SM89/SM90 Capability Gates

Tests skipped due to CUDA compute capability checks (e.g., `@unittest.skipIf(not SM90OrLater, ...)`):
- These test GENERAL functionality that happens to need SM90 on NVIDIA hardware
- XPU can run the same functionality without SM90
- Classify as `To be enabled`
- DetailReason: `Skipped due to SM90OrLater CUDA capability gate; XPU should support this test`

## Classification Examples (Proven Decisions)

### Closed skipIfXpu Issues

| Test | Issue | State | Classification |
|------|-------|-------|---------------|
| `test_copy_non_blocking_is_pinned_*` | #2334 | CLOSED 2026-03-25 | `To be enabled` |
| `test_div7_*` | intel-xpu-backend-for-triton#6401 | CLOSED 2026-04-22 | `To be enabled` |
| `test_comprehensive_nn_functional_linear_xpu_float16` | #2956 | CLOSED 2026-04-01 | `To be enabled` |

### Open Issues (Real Gaps)

| Test | Issue | State | Classification |
|------|-------|-------|---------------|
| `test_bad_cast` (fp8) | #2888 | OPEN | `Feature gap` |
| `gpu_cpp_wrapper` complex add tests | #3187 | OPEN | `Failures (xpu broken)` |
| `not-support-multithread` | #3098 | OPEN | `Feature gap` |

### Stale Skips Without Issues

| Test | Evidence | Classification |
|------|----------|---------------|
| `inductor_skips["xpu"]["lu"] = {f32}` | `torch.linalg.lu` passes through `torch.compile` on XPU | `To be enabled` |
| `inductor_skips["xpu"]["masked.cumprod"] = {f16}` | `torch.masked.cumprod` passes on XPU | `To be enabled` |
| `test_remove_noop_slice` compile_subprocess skip | Base test passes in `test_torchinductor.py` | `To be enabled` |

### Local Passed Examples

| Test | Evidence | Classification |
|------|----------|---------------|
| Slow test with `PYTORCH_TEST_WITH_SLOW=1` | Ran 1 test in 77.9s, OK | `Local Passed` |
| `test_low_memory_max_pool_dilation_*` | XPU variants pass (use_block_ptr_{False,True}_xpu) | `Local Passed` |

### CPU Tests (Not applicable)

| Test | Evidence | Classification |
|------|----------|---------------|
| `test_fp8_cpu` | Test name ends with `_cpu`, skip msg "requires GPU" | `Not applicable`, DetailReason=`CPU Case` |
| `test_while_loop_with_parameters_cpu` | Test name ends with `_cpu` | `Not applicable`, DetailReason=`CPU Case` |
| `test_fp8_cpu_with_stack_allocation` | Test name contains `_cpu_` | `Not applicable`, DetailReason=`CPU Case` |

### Environment Limitations

| Test | Evidence | Classification |
|------|----------|---------------|
| `Fails with Triton update` | Unconditional `unittest.skip` in source, all backends | `Test Enviroment limitation` |
| `GCC 13 vector codegen` | CPU test, compiler version dependent | `Test Enviroment limitation` |
| `Requires at least 2 GPUs` | Hardware requirement | `Test Enviroment limitation` |

## Output Rules

- `Reason`: use canonical workbook labels exactly as spelled
- `DetailReason`: include full issue/PR URL (e.g., `https://github.com/pytorch/pytorch/issues/NNNNN`),
  never bare numbers like `#NNNNN`. Extract URLs from `message_xpu` when present.
  Include test identity, tools used, evidence found, and reasoning chain.

## Verification

- Re-open output workbook with `openpyxl`
- Confirm 0 eligible blank `Reason` rows remain
- Confirm `Reason TBD` values unchanged
- Confirm updated cells are blue
- Spot-check at least one row from each classification category

## Working File

This subskill operates on the working file produced by the parent's extract +
optional filter steps. If a row filter was applied (e.g. the user said
"classify rows where `Reason='To be enabled' AND DetailReason='Daisy'`"), the
working file is a subset with a `_source_row` column. Edit only the subset's
`Reason` and `DetailReason`; write the verdicts back to the extracted file
(or to `.agent.xlsx`) via `classify_ut/scripts/apply_filtered_changes.py`.
See parent `classify_ut/SKILL.md` "Row-Level Filter" section.
