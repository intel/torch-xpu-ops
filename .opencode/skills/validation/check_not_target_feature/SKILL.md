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
    "evidence": [str],      # Specific matches. If a not_target/wontfix issue exists, list its number FIRST, e.g., "intel/torch-xpu-ops#3127 (CLOSED, not_target)", then the API/sheet/code citation, e.g., "Matched 'aten::_cudnn_rnn' in Not Applicable sheet"
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
python3 .opencode/skills/torch-xpu-ops-validation/scripts/list_not_applicable.py --json
```

**Match Extraction**:
- **Test Identity**: Match `test_name` against globs in the sheet (e.g., `test_fake_tensor_xpu test_cudnn_rnn_*`).
- **Code/Traceback APIs**: Look for `aten::*`, `torch.*` in the test body or traceback, and see if they exist in the `Operation/API` column (e.g., `aten::_cudnn_rnn`).
- **Error Keywords**: Match explicit strings in the error message (e.g., `"cuDNN"`, `"jiterator"`) against the sheet descriptions.

If a match is found:
- `is_not_target = True`, `verdict = "Not applicable"`. Evidence must cite the exact `Operation/API` entry. Then run Step 6 to attach a `not_target`/`wontfix` issue number if one exists.

### 4. Known "Not Target" Issues
Run parallel searches to check if the specific test or API was already closed as `not_target`/`wontfix` in `intel/torch-xpu-ops`:
```bash
gh search issues "<test_name_no_suffix> not_target is:issue" --repo=intel/torch-xpu-ops --limit=3 &
gh search issues "<test_name_no_suffix> wontfix is:issue" --repo=intel/torch-xpu-ops --limit=3 &
wait
```
If matched and verified via `gh issue view` (state=CLOSED, label=`not_target`|`wontfix`):
- `is_not_target = True`, `verdict = "Not applicable"`. Evidence must list the issue number and URL, e.g., `intel/torch-xpu-ops#3127 (CLOSED, not_target) https://github.com/intel/torch-xpu-ops/issues/3127`.

### 5. Implementation Analysis (Fallback)
If no authoritative source matched above, inspect the test implementation:
```bash
grep -A 20 -n "def ${test_name_no_suffix}" "$PYTORCH_SRC/${test_file}"
```
- **CUDA-Only Logic**: `torch.cuda.jiterator`, assertions on CUDA hardware internals (SM count, warp size).
- **Device-Agnostic with CUDA suffix**: Standard APIs (`torch.add`, `torch.Stream`) suffixed for CUDA. -> **Not not-target**.
- **JIT**: Owner-team scope (`torch.jit.*`, `oncall:jit`) is implicitly `Not applicable`.

If the test is strictly device-agnostic parametrization or just missing an implementation -> `is_not_target = False`.

### 6. Attach Issue Number (if possible)
Whenever the verdict is `Not applicable`, list the corresponding `intel/torch-xpu-ops` issue number in `evidence` if one exists -- even when the verdict was reached via the sheet (Step 3) or implementation analysis (Step 5), which do not inherently produce an issue.

- If Step 4 already matched a CLOSED `not_target`/`wontfix` issue, that number is the evidence -- done.
- Otherwise, run the Step 4 search now and verify any hit with `gh issue view`. If a CLOSED `not_target`/`wontfix` issue matches the test, prepend its number to `evidence` (format `intel/torch-xpu-ops#NNNN`).
- If no issue is found, keep the sheet/code citation as the evidence. The issue number is preferred but optional ("if possible"); never fabricate one.

## Strict Constraints (ZERO TOLERANCE)

1. **Default to Enablement**: If there is no explicit `not_target` label, no match in the Not-applicable sheet, and no strictly CUDA-hardware-tied logic, default to `is_not_target = False` (`To be enabled`).
2. **Missing Ops are NOT "Not Applicable"**: An error like `"is not implemented for xpu"` means it is missing (enablement gap), not out of scope.
3. **Parametrization gaps are NOT "Not Applicable"**: Missing `@dtypesIfXPU` when `@dtypesIfCUDA` exists is a gap, not a scope decision.
4. **Tool Restriction**: Use `bash` (with `gh` and python scripts), `read`, `grep`. No web tools. `gh search issues` must use `is:issue` to filter PRs out.
5. **No Blind Copies**: Do not copy input classification columns. Evaluate from scratch.
6. **Prefer Issue Numbers as Evidence**: When the verdict is `Not applicable` and a corresponding CLOSED `not_target`/`wontfix` issue exists, its number MUST appear in `evidence` (format `intel/torch-xpu-ops#NNNN`). Only fall back to a sheet `Operation/API` entry or a `file:line` code citation when no such issue is found. Never invent an issue number.