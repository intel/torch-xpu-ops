# `/quick-workflow` — Fast-Path Alternative

A stripped-down, three-stage pipeline that skips the review gate, environment
provisioning, and issue-filing phases. Use when you already know the target
class is device-generic and just want XPU on, verified, and a PR opened.

```
develop-xpu-test  ->  pytest + analyze-ut-results  ->  submit-xpu-test-pr
```

## When to Use

| Factor | `quick-workflow` | `enable-xpu-test` |
|--------|-----------------|-------------------|
| Review gate | ❌ Skipped — caller asserts clean | ✅ Full review |
| Provisioning | ❌ Env must already exist | ✅ Auto-provisions if missing |
| Issue filing | ❌ Failures reported only | ✅ Files issues via `create-xpu-issue` |
| Latent test bugs | Routes to `followup_classes` | Routes to `followup_classes` |
| Logging | `agent_space/quick_workflow/` | `agent_space/enable_xpu_orchestrator/` |

**Use `quick-workflow` when** you've already reviewed the test file (or it's
trivially device-generic like `TestDropoutNNDeviceType`) and you're confident
the only change needed is `allow_xpu=True`. The review gate in
`enable-xpu-test` would confirm 0 blockers anyway — `quick-workflow` saves that
phase.

**Use `enable-xpu-test` when** the file hasn't been reviewed yet, the env might
need setup, or you want automatic issue filing for failures.

## Quick Start

### Prerequisites

Same as `enable-xpu-test` — conda env, pytorch checkout, `gh` auth — but they
must already exist. No auto-provisioning.

### One-Line Command

```
/quick-workflow <test_file>::<ClassName>  conda_env=<env>  pytorch_folder=<path>
```

Multiple classes, multiple files:

```
/quick-workflow nn/test_dropout.py::TestDropoutNNDeviceType \
    test/distributions/test_distributions.py::TestDistributions \
    conda_env=classify_ut_test pytorch_folder=~/daisy_pytorch
```

## Walkthrough (Same Example, Fewer Steps)

Enabling `TestDropoutNNDeviceType` with `quick-workflow` produces the same
edit and the same PR, but skips the review gate and provisioning check.

### Stage 1: Develop

Same `develop-xpu-test` subskill, same three edit types. For
`TestDropoutNNDeviceType` only Type 1 is needed:

```diff
-instantiate_device_type_tests(TestDropoutNNDeviceType, globals(), allow_mps=True)
+instantiate_device_type_tests(TestDropoutNNDeviceType, globals(), allow_mps=True, allow_xpu=True)
```

No review gate call — the subagent goes straight to editing. If it discovers a
class has no device axis at all (e.g. a plain `unittest.TestCase`), it skips
that class and reports why rather than forcing an edit.

Output artifact: `agent_space/quick_workflow/stage1_develop.json`

### Stage 2: Verify + Analyze

Runs pytest for the XPU variants of all classes in the file group:

```bash
pytest test/nn/test_dropout.py -k "TestDropoutNNDeviceType and xpu" \
    --timeout 600 --tb=line -q
```

```
TestDropoutNNDeviceTypeXPU::test_Dropout_xpu          PASSED
TestDropoutNNDeviceTypeXPU::test_Dropout1d_xpu        PASSED
TestDropoutNNDeviceTypeXPU::test_Dropout2d_xpu        PASSED
TestDropoutNNDeviceTypeXPU::test_Dropout3d_xpu        PASSED
TestDropoutNNDeviceTypeXPU::test_empty_dropout_xpu    PASSED
5 passed in 2.60s
```

**On failure** (unlike `enable-xpu-test`): Instead of filing a GitHub issue,
the workflow either:
- Adds an `@expectedFailureXPU` decorator (if the root cause is a tracked
  upstream issue, verified OPEN via `gh issue view`)
- Routes the class to `followup_classes` with a bug description (if it's a
  latent bug in test body logic like missing `device=`)
- The `analyze-ut-results` subskill groups failures by error signature and
  cross-references known issues

### Stage 3: Submit

Identical to `enable-xpu-test` — calls `submit-xpu-test-pr` once for all
passing classes:

```
PR: https://github.com/pytorch/pytorch/pull/189254
Branch: xpu/enable-test-dropout
Base: viable/strict
Draft: yes (confirm-gated)
```

## Input Reference

### Positional

`test_file::ClassName` pairs, space-separated:

```
/quick-workflow <file1>::<Class1> [<file2>::<Class2> ...] \
    conda_env=<env> pytorch_folder=<path>
```

### Named

| Field | Required | Description |
|-------|----------|-------------|
| `conda_env` | Yes | Existing conda env with XPU PyTorch. Must exist — not created here. |
| `pytorch_folder` | Yes | Existing local pytorch checkout. Must exist — not cloned here. |

Complex multi-class input (JSON):

```json
{
  "test_targets": [
    {"test_file": "nn/test_dropout.py", "test_class": "TestDropoutNNDeviceType"},
    {"test_file": "test/distributions/test_distributions.py", "test_class": "TestDistributions"}
  ],
  "conda_env": "classify_ut_test",
  "pytorch_folder": "/home/daisyden/daisy_pytorch"
}
```

## Output Format

```json
{
  "status": "submitted|partial|no-passing-classes",
  "pr_url": "https://github.com/pytorch/pytorch/pull/189254",
  "passed_targets": ["nn/test_dropout.py::TestDropoutNNDeviceType"],
  "followup_targets": [
    {"test_file": "...", "test_class": "...", "root_cause": "..."}
  ],
  "per_target": [
    {
      "test_file": "nn/test_dropout.py",
      "test_class": "TestDropoutNNDeviceType",
      "pytest_summary": "5 passed, 0 failed, 0 skipped",
      "outcome": "enabled"
    }
  ]
}
```

### `status` Values

| Status | Meaning |
|--------|---------|
| `submitted` | All passed, PR opened |
| `partial` | Some passed (PR opened), some in followup |
| `no-passing-classes` | All classes ended up in followup; no PR |

## Logging Artifacts

All under `agent_space/quick_workflow/` (auto-backed up each run):

```
agent_space/quick_workflow/
├── session_log.txt                  # One-line-per-event timeline
├── stage1_develop.json              # Per-file-group edit summary
├── stage2_verify_analyze.json       # Pytest results + analyze verdicts
├── stage3_submit.json               # PR submission result
└── logs/
    └── <file_group_slug>_pytest.log # Raw pytest output (appended)
```

Each run backs up the previous `quick_workflow/` directory with a UTC
timestamp before creating a fresh one. Nothing is silently overwritten.

## Architecture

```
User command
    │
    ▼
┌────────────────────────────────────────────┐
│         quick-workflow (orchestrator)       │
│  Stages: develop → verify+analyze → submit  │
│  No review gate, no provisioning, no issues  │
└──┬───┬───┬──────────────────────────────────┘
   │   │   │
   ▼   ▼   ▼
   │   │   │
   ├── develop-xpu-test         (Stage 1)
   ├── pytest + analyze-ut-results  (Stage 2)
   └── submit-xpu-test-pr       (Stage 3)
```

The workflow reuses subagent `task_id` within a file group so the file is
read/edited exactly once.

## Constraints

1. **No review gate.** Caller asserts the file is device-generic. If
   `develop-xpu-test` discovers a class has no device axis, it skips that class
   and reports why.
2. **No environment provisioning.** Missing/broken preconditions are a hard
   stop, not auto-provisioned.
3. **No issue filing.** Failures are handled with `@expectedFailureXPU`
   markers or routed to `followup_classes`. No GitHub issues are created.
4. **Never edit test method body logic.** Even a one-token fix (like adding
   `device=`) is out of scope. Route the class to `followup_classes` instead.
5. **Never cite a closed GitHub issue as gating justification.** Check issue
   state via `gh issue view`; only OPEN counts.
6. **A class is only `passed` when its final run shows 0 unhandled FAILED.**
   xfailed/skipped is clean. An `expectedFailure` that passes must be reverted.
7. **PR submission is confirm-gated.** No `git commit`, `git push`, or
   `gh pr create` without explicit user approval.
8. **Logging is mandatory and session-scoped.** Every run starts by backing
   up the previous `quick_workflow/` directory (never silently overwritten).
