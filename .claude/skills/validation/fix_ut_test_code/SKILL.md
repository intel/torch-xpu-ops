---
name: fix-ut-test-code
description: Apply allowed test-code fixes to a failing XPU unit test (sys.path additions, CUDA-to-XPU API generalization, missing-attribute skip guards, syntax fixes), then rerun to confirm. Returns fixed-and-passing, or escalates to issue submission when the root cause is not test code.
---

# `fix-ut-test-code`

## Purpose

Given a failing test group classified as `test-code` by `analyze-ut-results`,
apply the minimal allowed fix and rerun. Confirm the fix resolves the failure,
or conclude the root cause is NOT test code and hand off to `create-xpu-issue`.

## Input

- `tests`: failing `TestClass.test_name` entries and their file paths.
- `root_cause`: the hypothesis from `analyze-ut-results`.
- `conda_env`, `pytorch_root`: the environment and checkout established by the
  calling agent.

## Environment

Use the `conda_env` and `pytorch_root` passed in by the caller. **Do NOT set up
or activate any environment** (no `setup_env.sh`). Run pytest inside the
checkout with the provided env, e.g. `conda run -n "${conda_env}" pytest ...`
from `${pytorch_root}/third_party/torch-xpu-ops/test/xpu`.

## Allowed vs prohibited modifications

| Fix type | Allowed |
|---|---|
| Import path / `sys.path` additions | Yes |
| CUDA -> XPU API generalization | Yes |
| Missing-attribute skip guards | Yes |
| f-string / syntax error fixes | Yes |
| Changing test expectations or assertions | No |
| Modifying backend / infrastructure code | No |
| Removing test coverage | No |
| Weakening skip conditions | No |

Every changed line must trace directly to making the test correctly run or
skip on XPU. Do not "improve" adjacent code.

## Fix patterns

**1. Import path missing** — make pytorch `test/dynamo` importable:

```python
from pathlib import Path
PYTORCH_DYNAMO_PATH = str(Path(__file__).resolve().parents[5] / "test" / "dynamo")
if os.path.exists(PYTORCH_DYNAMO_PATH) and PYTORCH_DYNAMO_PATH not in sys.path:
    sys.path.insert(0, PYTORCH_DYNAMO_PATH)
```

**2. CUDA-specific API** — make the call device-aware:

```python
s = torch.xpu.Stream(device=GPU_TYPE) if GPU_TYPE == "xpu" else torch.cuda.Stream(device=GPU_TYPE)
```

**3. Missing attribute on XPU object** — guard with a citing skip:

```python
cs = torch.xpu.current_stream()
if not hasattr(cs, "cuda_stream"):
    self.skipTest("cuda_stream attribute not available on XPU stream")
```

## Workflow

1. Apply the smallest fix matching `root_cause`.
2. Rerun only the affected test(s):

```bash
cd $HOME/daisy_pytorch/third_party/torch-xpu-ops/test/xpu
pytest -v --timeout=120 dynamo/test_<name>_xpu.py -k "<test_pattern>"
```

3. Evaluate:
   - **Passes** -> fix succeeded.
   - **Skips with a citing reason** -> acceptable for a parametrization/attribute gap.
   - **Still fails** -> root cause is NOT test code. Revert your edit and escalate.

```
Is the failure due to a test-code bug?
- Import path missing       -> add sys.path, rerun
- CUDA->XPU not generalized  -> generalize API, rerun
- Missing attribute         -> add skip guard, rerun
- Otherwise                 -> revert, hand off to create-xpu-issue
```

## Output

Return JSON:

```json
{
  "tests": ["TestClass.test_a"],
  "outcome": "fixed-passing|fixed-skipping|escalate-to-issue",
  "fix_type": "sys.path|cuda-to-xpu|skip-guard|syntax|none",
  "files_changed": ["test/xpu/dynamo/test_x_xpu.py"],
  "rerun_command": "pytest -v ...",
  "escalation_reason": "present only when outcome == escalate-to-issue"
}
```

## Constraints

- **Tools**: `bash` (`pytest`, `git`), `read`, `edit`, `grep`.
- Only the allowed modifications above. Never touch backend/infra code or
  test assertions.
- If a fix does not make the test pass or cleanly skip, revert it before
  escalating — leave the tree clean.
- Add an inline `# Tracked in intel/torch-xpu-ops#NNNN` comment next to any
  skip guard once the tracking issue exists (issue is filed by `create-xpu-issue`).
