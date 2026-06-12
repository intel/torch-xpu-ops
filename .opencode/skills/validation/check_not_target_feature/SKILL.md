---
name: check-not-target-feature
description: Check whether a given test case is a "not target" feature for XPU — i.e., whether it should be classified as Not applicable because it tests CUDA-only behavior that is explicitly out of scope for XPU.
---

# `check-not-target-feature`

## Objective
Determine whether a test is out-of-scope for XPU (e.g., testing CUDA-specific internal behavior, cuDNN-specific APIs) and should be classified as `Not applicable`, or if it is a standard PyTorch test that XPU should eventually support (`To be enabled`).

## Inputs
- `test_file`, `class_name`, `test_name`, `device`
- `error_message`, `test_code_block`, `traceback` (if provided)

## Output Format
Return this JSON object:
```python
{
    "is_not_target": bool,
    "verdict": "Not applicable" | "Not not-target" | "CPU Case",
    "evidence": [str],      # Explain specific matches, e.g., "Matched 'aten::_cudnn_rnn' in Not Applicable sheet"
    "reasoning": str        # Brief summary of the decision
}
```

## Deep Analysis Workflow

### 1. Mandatory Input Scrubbing
- **Ignore** any pre-existing `Reason` or `DetailReason` from the task input. Do not carry them forward.
- **Never read the Excel file directly.** 

### 2. CPU-Only Fast Path
If the test name ends with `_cpu`, contains `_cpu_`, or the skip message explicitly says "requires GPU" but targets CPU:
- `is_not_target = False`, `verdict = "CPU Case"`

### 3. "Not Applicable" Sheet Check (Authoritative)
The sheet at `https://github.com/daisyden/ai_for_validation/blob/main/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx` is the absolute source of truth.

Run the bundled script to fetch the current JSON export of this sheet:
```bash
python3 .opencode/skills/classify_ut/scripts/list_not_applicable.py --json
```

**Match Extraction**:
- **Test Identity**: Match `test_name` against globs in the sheet (e.g., `test_fake_tensor_xpu test_cudnn_rnn_*`).
- **Code/Traceback APIs**: Look for `aten::*`, `torch.*` in the test body or traceback, and see if they exist in the `Operation/API` column (e.g., `aten::_cudnn_rnn`).
- **Error Keywords**: Match explicit strings in the error message (e.g., `"cuDNN"`, `"jiterator"`) against the sheet descriptions.

If a match is found:
- `is_not_target = True`, `verdict = "Not applicable"`, Evidence must cite the exact `Operation/API` entry.

### 4. Known "Not Target" Issues
Run parallel searches to check if the specific test or API was already closed as `not_target`/`wontfix` in `intel/torch-xpu-ops`:
```bash
gh search issues "<test_name_no_suffix> not_target is:issue" --repo=intel/torch-xpu-ops --limit=3 &
gh search issues "<test_name_no_suffix> wontfix is:issue" --repo=intel/torch-xpu-ops --limit=3 &
wait
```
If matched and verified via `gh issue view` (state=CLOSED, label=`not_target`|`wontfix`):
- `is_not_target = True`, `verdict = "Not applicable"`, Evidence must cite the issue URL.

### 5. Implementation Analysis (Fallback)
If no authoritative source matched above, inspect the test implementation:
```bash
grep -A 20 -n "def ${test_name_no_suffix}" "$PYTORCH_SRC/${test_file}"
```
- **CUDA-Only Logic**: `torch.cuda.jiterator`, assertions on CUDA hardware internals (SM count, warp size).
- **Device-Agnostic with CUDA suffix**: Standard APIs (`torch.add`, `torch.Stream`) suffixed for CUDA. -> **Not not-target**.
- **JIT**: Owner-team scope (`torch.jit.*`, `oncall:jit`) is implicitly `Not applicable`.

If the test is strictly device-agnostic parametrization or just missing an implementation -> `is_not_target = False`.

## Strict Constraints (ZERO TOLERANCE)

1. **Default to Enablement**: If there is no explicit `not_target` label, no match in the Not-applicable sheet, and no strictly CUDA-hardware-tied logic, default to `is_not_target = False` (`To be enabled`).
2. **Missing Ops are NOT "Not Applicable"**: An error like `"is not implemented for xpu"` means it is missing (enablement gap), not out of scope.
3. **Parametrization gaps are NOT "Not Applicable"**: Missing `@dtypesIfXPU` when `@dtypesIfCUDA` exists is a gap, not a scope decision.
4. **Tool Restriction**: Use `bash` (with `gh` and python scripts), `read`, `grep`. No web tools. `gh search issues` must use `is:issue` to filter PRs out.
5. **No Blind Copies**: Do not copy input classification columns. Evaluate from scratch.