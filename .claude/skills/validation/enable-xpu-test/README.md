# `/enable-xpu-test` — User Guide

Enable XPU backend coverage on PyTorch test classes that are already
device-generic (accelerator-agnostic), then open a single draft PR against
`pytorch/pytorch`.

## What It Does

The orchestrator runs these phases for each `test_file` group:

```
Phase 0.5:  Provision conda env + pytorch checkout (if missing)
Phase 1:    Review gate — is the test class ready for XPU?
Phase 2:    Develop — apply the enablement edits
Phase 2.5:  Remove skips — probe and remove stale method-level XPU skips (optional)
Phase 3:    Verify — run the enabled tests on XPU
Phase 4:    Analyze — only if tests had failures (else skip)
Phase 5A:   Submit — open ONE combined draft PR for all passing classes
Phase 5B:   Follow-up — resolve failing classes: known-issue → `@skipIfXpu`; else file new issue → revert
```

Classes that fail review or verification are **isolated** — they don't block
siblings. Only batch-wide errors (broken env, missing inputs) hard-stop.

## Quick Start

### Prerequisites

- A conda environment with a PyTorch XPU wheel installed
- A local `pytorch/pytorch` checkout (the test file lives there)
- `gh` authenticated with a token that can push to `daisyden/pytorch`

### One-Line Command

```
/enable-xpu-test <test_file>  test_class=<ClassName>  conda_env=<env>  pytorch_folder=<path>
```

Or with the class inline:

```
/enable-xpu-test nn/test_dropout.py::TestDropoutNNDeviceType  conda_env=my_env  pytorch_folder=~/pytorch
```

## Walkthrough (Real Example)

Let's walk through enabling `TestDropoutNNDeviceType` from
`test/nn/test_dropout.py`.

### Step 1: Understand the Target

Before running, check the test file:

```python
# test/nn/test_dropout.py (line 326)
instantiate_device_type_tests(TestDropoutNNDeviceType, globals(), allow_mps=True)
```

The class already:
- Accepts a `device` parameter in every test method
- Uses only generic PyTorch ops (no `torch.cuda.*`, no `.cuda()`, no `@onlyCUDA`)
- Has no hardcoded `device="cuda"` strings

That means it's **Strategy 2** (device-agnostic) — exactly what this
orchestrator handles.

### Step 2: Invoke the Orchestrator

```
/enable-xpu-test nn/test_dropout.py  test_class=TestDropoutNNDeviceType  \
    conda_env=classify_ut_test  pytorch_folder=~/daisy_pytorch
```

### Step 3: What Happens (Phase by Phase)

#### Phase 0.5 — Provisioning (conditional)

If the conda env or pytorch checkout is missing, `setup_env.sh` is called:

```bash
bash .opencode/skills/validation/scripts/setup_env.sh \
    nightly "classify_ut_test" "/home/daisyden/daisy_pytorch"
```

If both already exist (as in this run), provisioning is skipped and the
orchestrator logs:

```
[2026-07-08 10:10:46] phase_0.5_provision | env: exists | pytorch: exists
```

#### Phase 1 — Review Gate

The test file is reviewed by the `review-test-refactoring` skill. It checks:

| Check | What It Looks For |
|-------|-------------------|
| Classification | Is the class really Strategy 2? |
| CUDA hardening | Any `@onlyCUDA`, `.cuda()`, `device="cuda"`? |
| Naming | Does the class name follow `FooDeviceType` convention? |
| API usage | Any `torch.cuda.*` that should be `torch.accelerator.*`? |
| XPU-specific | Any bfloat16 guards, dtype issues, memory_format concerns? |

**For `TestDropoutNNDeviceType`:** The review found **0 blockers**. The only
finding was a minor one: missing `allow_xpu=True` on line 326. All test
methods are device-generic, no CUDA-specific APIs, no issues with
`channels_last` or bfloat16.

Output artifact: `agent_space/enable_xpu_orchestrator/phase1_review.json`

```json
{
  "results": [{
    "test_class": "TestDropoutNNDeviceType",
    "verdict": "pass",
    "blockers": 0
  }],
  "eligible_classes": ["TestDropoutNNDeviceType"]
}
```

#### Phase 2 — Develop Enablement

The `develop-xpu-test` skill applies up to **three possible edit types**:

1. **Instantiation enablement** — always inline `only_for=("cpu", "cuda", "xpu")`, never a separate variable.
2. **Decorator parity** — mirror CUDA-only decorators to XPU (none needed).
3. **op_db widening** — extend `DecorateInfo` entries in
   `common_methods_invocations.py` (none matched → no change).

For a simple case like `TestDropoutNNDeviceType`, only edit type 1 is needed:

```diff
-instantiate_device_type_tests(TestDropoutNNDeviceType, globals(), allow_mps=True)
+instantiate_device_type_tests(TestDropoutNNDeviceType, globals(), allow_mps=True, allow_xpu=True)
```

That's it — 1 insertion, no test code touched, no skips needed.

#### Phase 2.5 — Remove Stale Skips (optional)

After enablement, some methods may still be skipped by individual decorators
(`@skipIfXpu`, `@skipXPU`, `@skipXPUIf`) or inline device-type guards
(`if self.device_type not in ("cpu", "cuda"): self.skipTest(...)`). The
`remove-xpu-skips` skill probes these one at a time:

1. **Discover** all skips in the enabled class.
2. **Check issue** (for decorators with a GitHub issue URL): skip removal if
   the referenced issue is OPEN; proceed only if CLOSED.
3. **Try removal**: remove the decorator or widen the guard.
4. **Run the test** on XPU.
5. **Keep or revert**: pass → keep; fail → revert.

For `TestChebyshevNanPropagation`, `remove-xpu-skips` found one P5 guard
(`"cpu", "cuda"`) and attempted to widen it to include `"xpu"`. The test had
a pre-existing NaN propagation subfailure, so the change was **reverted** —
the guard stays as-is. The class-level enablement (Phase 2) was kept.

Output artifacts: `agent_space/remove_xpu_skips/` (per-skip report + raw
pytest logs).

#### Phase 3 — Verify on XPU

The `verify-xpu-test` skill runs the enabled class on the XPU host:

```bash
conda run -n classify_ut_test python -m pytest \
    test/nn/test_dropout.py -v -k "TestDropoutNNDeviceType" --tb=short
```

You see output like:

```
TestDropoutNNDeviceTypeCPU::test_Dropout_cpu          PASSED
TestDropoutNNDeviceTypeCPU::test_Dropout1d_cpu        PASSED
TestDropoutNNDeviceTypeCPU::test_Dropout2d_cpu        PASSED
TestDropoutNNDeviceTypeCPU::test_Dropout3d_cpu        PASSED
TestDropoutNNDeviceTypeCPU::test_empty_dropout_cpu    PASSED
TestDropoutNNDeviceTypeXPU::test_Dropout_xpu          PASSED
TestDropoutNNDeviceTypeXPU::test_Dropout1d_xpu        PASSED
TestDropoutNNDeviceTypeXPU::test_Dropout2d_xpu        PASSED
TestDropoutNNDeviceTypeXPU::test_Dropout3d_xpu        PASSED
TestDropoutNNDeviceTypeXPU::test_empty_dropout_xpu    PASSED
```

Verdict: **verified** — 5 CPU + 5 XPU tests, all passed.

#### Phase 4 — Analyze (only on failure)

For all-pass classes, `verify-xpu-test` counts are used directly as the verdict
(`passed`). These classes advance to the PR.

For classes with failures, `analyze-ut-failures` groups each failure by root
cause. The class's edits are **kept in the tree** (not reverted yet) and moved
to Phase 5B for resolution: known-issue check → optional `@skipIfXpu` addition
or new issue filing.

#### Phase 5 — Submit PR

When all classes pass, a single draft PR is opened:

```bash
gh pr create \
  --repo pytorch/pytorch \
  --base viable/strict \
  --head daisyden:xpu/enable-test-dropout \
  --draft \
  --title "[XPU][Test] Enable TestDropoutNNDeviceType on XPU"
```

PR: https://github.com/pytorch/pytorch/pull/189254

| Detail | Value |
|--------|-------|
| Branch | `xpu/enable-test-dropout` |
| Base | `viable/strict` (upstream `pytorch/pytorch`) |
| Files changed | 1 (`test/nn/test_dropout.py` — +1/-1) |
| Draft? | Yes — confirm-gated before submission |
| Fork | `daisyden/pytorch` |

If a class had failures instead, Phase 5B would:
1. **Classify**: test-code bug → revert; backend gap → proceed.
2. **`check-known-issue`** — is each failure already tracked?
3. **Known issue → `@skipIfXpu`**: add the decorator on individual failing
   methods with the existing issue URL. Class stays enabled (skipped methods
   don't run, passing siblings do).
4. **No known issue → `create-xpu-issue`**: file a structured issue. No skip
   added (no URL to reference). Class is reverted.
5. **Gather issue URLs** for PR body cross-linking.

## Input Reference

### Positional

| Field | Required | Example |
|-------|----------|---------|
| `test_file` | Yes | `nn/test_dropout.py` |

### Named

| Field | Required | Example |
|-------|----------|---------|
| `test_class` | Yes | `TestDropoutNNDeviceType` |
| `conda_env` | Yes | `classify_ut_test` |
| `pytorch_folder` | Yes | `~/daisy_pytorch` |

The `test_class` can be omitted from named params by inlining it in the
`test_file`:

```
/enable-xpu-test nn/test_dropout.py::TestDropoutNNDeviceType
```

## Output Format

The orchestrator returns a JSON summary:

```json
{
  "status": "passed",
  "pytorch_folder": "/home/daisyden/daisy_pytorch",
  "pr_url": "https://github.com/pytorch/pytorch/pull/189254",
  "passed_targets": ["nn/test_dropout.py::TestDropoutNNDeviceType"],
  "pending_targets": [],        # Failures pending Phase 5B resolution
  "followup_targets": [],       # Reverted + new issues filed (5B dead end)
  "known_issue_urls": [],
  "created_issue_urls": [],
  "per_target": [
    {
      "test_file": "nn/test_dropout.py",
      "test_class": "TestDropoutNNDeviceType",
      "review": "pass",
      "verify": "verified",
      "analysis_verdict": "passed",
      "outcome": "enabled"
    }
  ]
}
```

### `status` Values

| Status | Meaning |
|--------|---------|
| `passed` | All classes enabled, PR opened |
| `partial` | Some enabled (PR opened), rest in `pending_targets` → Phase 5B resolved (skipped or reverted) |
| `issue-follow-up` | None enabled, only issues filed (no PR) |
| `failed-hard-stop` | Batch-wide critical error (env, auth, missing inputs) |

## Logging Artifacts

All logs go under `agent_space/` in the `torch-xpu-ops` checkout:

```
agent_space/
├── session_log.txt                          # Human-readable timeline
├── logs/
│   └── background_status.log                # Subagent status
├── remove_xpu_skips/                        # Phase 2.5 skip-removal artifacts
│   ├── <test_slug>__<class>__discovery.json
│   ├── <test_slug>__<class>.json
│   └── logs/
│       └── <test_slug>__<class>__<method>__xpu.txt
└── enable_xpu_orchestrator/
    ├── phase1_review.json
    ├── phase2_develop.json
    ├── phase3_verify.json
    ├── phase4_analyze.json                  # Only on failure
    ├── phase5_followup.json                 # Only on failure
    └── phase5_submit_pr.json
```

Each phase JSON is keyed by file-group slug with a `per-class` results array.

## Common Scenarios

### Scenario A: All Tests Pass (Happy Path)

```
/enable-xpu-test nn/test_dropout.py test_class=TestDropoutNNDeviceType \
    conda_env=classify_ut_test pytorch_folder=~/daisy_pytorch
```

→ PR opened for all classes. Took ~15 minutes.

### Scenario B: Some Tests Fail (Partial Enablement)

If `TestA` passes but `TestB` fails with a backend gap:

- `TestA` → verified → PR
- `TestB` → Phase 5B resolves:
  - **Known issue already tracked** → `@skipIfXpu` on failing methods only,
    class stays enabled — passing siblings still run on XPU
  - **No known issue** → `create-xpu-issue` filed, class **reverted**,
    moved to `followup_targets`
- PR body links to known issue URLs and/or filed issues

### Scenario C: Review Gate Blocks a Class

If the class still has `@onlyCUDA` or `.cuda()` calls:

```
/enable-xpu-test test/test_foo.py test_class=TestFoo \
    conda_env=my_env pytorch_folder=~/pytorch
```

→ Blockers reported. Class skipped. No PR (zero passing classes). Fix the
blockers, then re-run.

### Scenario D: Environment Needs Setup

```
/enable-xpu-test nn/test_dropout.py test_class=TestDropoutNNDeviceType \
    conda_env=new_env pytorch_folder=~/new_pytorch
```

→ `setup_env.sh` creates the conda env and clones the pytorch checkout.
If it fails (e.g., no internet, broken driver), the orchestrator hard-stops
with a fatal error.

## Architecture

```
User command
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│              enable-xpu-test (orchestrator)                │
│  Phases: provision → review → develop → remove-skips →    │
│          verify → analyze → 5A: submit PR / 5B: follow-up │
│  Group-by-file, batch per phase, one PR at end             │
└──┬───┬───┬───┬───┬───┬───┬────────────────────────────────┘
   │   │   │   │   │   │   │
   ▼   ▼   ▼   ▼   ▼   ▼   ▼
   │   │   │   │   │   │   │
   ├── review-test-refactoring    (Phase 1)
   ├── develop-xpu-test           (Phase 2)
   ├── remove-xpu-skips           (Phase 2.5, optional)
   ├── verify-xpu-test            (Phase 3)
   ├── analyze-ut-failures        (Phase 4, only if needed)
   ├── check-known-issue          (Phase 5B)
   ├── create-xpu-issue           (Phase 5B)
   └── submit-xpu-test-pr         (Phase 5A)
```

`remove-xpu-skips` is an optional Phase 2.5 subskill. Use it after develop
enablement to clean up stale method-level skips whose tracking issues have
been resolved. Each skip is probed individually: issue check → removal →
pytest → keep/revert. Kept removals stay in the working tree alongside
Phase 2's enablement edits and proceed together to Phase 3 verification.

Each subskill is a focused agent that handles one concern. The orchestrator
reuses subagent sessions via `task_id` within a file group to avoid redundant
file reads.

## Constraints (Non-Negotiable)

1. **`@skipIfXpu` is allowed only when backed by a tracking issue.** Added by
   Phase 5B after `check-known-issue` returns a match. No `@skipXPU`,
   `@skipXPUIf`, `self.skipTest("xpu", ...)`, or inline device-conditionals.
2. **Never edit test method body logic.** Even a one-token fix is out of scope.
3. **No op_db changes for other classes.** Only widen `DecorateInfo` entries
   belonging to the target class and its generic test names.
4. **No closed issues as gating justification.** Check issue state; only OPEN
   counts.
5. **Never skip verification.** Every enabled class must run on XPU first.
6. **Surgical revert only.** A failing class's edits are reverted without
   discarding passing siblings' changes (applies to test-code bugs; backend
   gaps use Phase 5B `@skipIfXpu` instead).
7. **PR is always a draft.** Confirm-gated — never publishes without approval.
