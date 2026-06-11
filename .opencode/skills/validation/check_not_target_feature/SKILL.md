---
name: check-not-target-feature
description: Check whether a given test case is a "not target" feature for XPU — i.e., whether it should be classified as Not applicable because it tests CUDA-only behavior that is explicitly out of scope for XPU.
---

# check_not_target_feature

Check whether a given test case is a "not target" feature for XPU — i.e., whether it should be classified as `Not applicable` because it tests CUDA-only behavior that is explicitly out of scope for XPU.

Derived from the `classify_ut` classification rules, specifically the **CUDA-Only Judgement Rule** and the **Sibling-Class Verdict Mapping** from `classify_ut/RULES.md`.

## Purpose

Given a test file, class name, and test name (optionally with device type), determine whether the test is a not-target feature for XPU. Returns verdict with evidence.

## Inputs

| Parameter | Description | Default |
|-----------|-------------|---------|
| `test_file` | Test file path relative to `$PYTORCH_SRC` (e.g. `test/dynamo/test_streams.py`) | **Required** |
| `class_name` | Test class name (e.g. `TestStreamsCUDA`) | **Required** |
| `test_name` | Test method name (e.g. `test_local_stream_enter_exit`) | Optional; when omitted, check the whole class |
| `device` | Device type: `cuda` or `xpu` | `cuda` |
| `error_message` | Error message from test failure (e.g. CUDA error, missing operator). Helps identify CUDA-specific behavior. | `None` |
| `test_code_block` | Source code of the test method body, from `xpu_test_knowledge` exploration. Used to match API/operator usage against the Not applicable sheet. | `None` |
| `traceback` | Full traceback from the test failure. Stack frames often name the failing operator (e.g. `aten::_cudnn_rnn`), which can be matched against Not applicable sheet entries. | `None` |
| `PYTORCH_SRC` | PyTorch source checkout path | `$HOME/upstream/pytorch` |

## Output

```python
{
    "is_not_target": bool,        # True if the test is not-target for XPU
    "verdict": str,               # "Not applicable" or "Not not-target" or "Need investigation"
    "confidence": str,            # "HIGH", "MEDIUM", or "LOW"
    "evidence": [                  # List of evidence strings
        "CUDA-only API: torch.cuda.jiterator",
        "Not-applicable sheet match: Issue #2918 (CUDA-specific implementation)",
    ],
    "reasoning": str,             # Free-text explanation
    "suggested_reason": str,      # "Not applicable", "To be enabled", "Feature gap", etc.
}
```

## Decision Flow

To minimize execution turns and lower latency, you MUST run independent data-gathering tool calls (e.g., `bash`, `read`, `grep`) concurrently using parallel tool invocations.

### Step 1: Check CPU-only test

If the test is CPU-only (test name ends with `_cpu`, contains `_cpu_`, skip message says "requires GPU"), the test is `Not applicable (CPU Case)` but NOT "not target" — it simply doesn't involve XPU.

- Result: `is_not_target = False`, `verdict = "CPU Case — not an XPU target decision"`

### Step 2: Check the "Not applicable" sheet (AUTHORITATIVE)

The **only** authoritative source for "out-of-XPU-scope behavior" is the `Not applicable` sheet of `torch_xpu_ops_issues.xlsx`. Its `Operation/API` column enumerates operators/APIs that are out of scope.

Source of truth:
```
https://github.com/daisyden/ai_for_validation/blob/main/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx
  └─ sheet: "Not applicable"
     └─ column: "Operation/API"  (with Issue ID | Category | State for citation)
```

Run the bundled script to fetch and check the list:

```bash
python3 .opencode/skills/classify_ut/scripts/list_not_applicable.py --json
```

Then check if any `Operation/API` entry matches the operator, API, error, or code pattern in the test. Use **all available inputs** — not just the test name:

**Signal sources to match against the sheet:**

| Input Source | What to extract | How to extract |
|---|---|---|
| `test_file` + `class_name` + `test_name` | Test identity patterns | Used directly — entries like `(test_fake_tensor_xpu test_cudnn_rnn_*)` match by test name glob |
| `test_code_block` (if provided) | Operator/API calls in the test body | Grep for `torch.*(`, `aten::*`, `torch.xpu.*`, `torch.cuda.*`, `torch.backends.*` |
| `error_message` (if provided) | Error strings that match known Not-applicable entries | Match error text against `Operation/API` descriptions — e.g., `"cuDNN"` in error → `aten::_cudnn_rnn` entry |
| `traceback` (if provided) | Operator names in stack frames | Frame lines like `"torch/cuda/"`, `"aten::"`, `"torch.xpu."` — these name the exact failing operator |

**Extraction patterns (illustrative — use your own tools to apply them):**

Write each input to a temp file, then run parallel `grep` tool calls or `bash` grep to extract symbols:

```
Pattern for API calls from test code block:
  (torch\.[a-zA-Z_.]+|aten::[a-zA-Z_]+)

Pattern for operator names from traceback:
  (aten::[a-zA-Z_]+|torch\.[a-zA-Z_.]+)

Pattern for error message against known Not-applicable entries:
  (cu[dD][nN][nN]|cu[bB][lL][aA][sS]|nccl|cuda\s+error|cuda\s+runtime|jit|cusparse)
```

**Match criteria:**
- Match extracted symbols (operator names, API calls) against `Operation/API` entries
- Match the raw `error_message` text against `Operation/API` descriptions or comments
- Match test identity (`test_file`/`class_name`/`test_name`) against any test-name globs listed in parentheses in the sheet entry
- An entry like `aten::_cudnn_rnn (test_fake_tensor_xpu test_cudnn_rnn_*)` matches by both the operator AND the test name pattern
- An entry like `torch.xpu.*Storage; TypedStorage.xpu (...)` matches by API symbol

**If match found:** `is_not_target = True`, `verdict = "Not applicable"`, `confidence = "HIGH"`
**If no match:**

If `test_code_block`, `error_message`, or `traceback` contain CUDA-specific operators/APIs but none match the Not-applicable sheet → this is likely an **enablement gap**, not a scope exclusion. Pop a warning in evidence and continue to Step 3.

**If no match and no CUDA-specific signals found:** Continue to Step 3.

### Step 3: Check for `not_target` / `wontfix` labels on known issues

Search for the test name, class name, or related APIs in issues with `not_target` or `wontfix` labels. The three searches are independent — run them in parallel:

```bash
gh search issues "<test_name> xpu not_target" --repo=intel/torch-xpu-ops --limit=10 &
gh search issues "<test_name> wontfix" --repo=intel/torch-xpu-ops --limit=10 &
gh search issues "<API_name> xpu not_target" --repo=intel/torch-xpu-ops --limit=10 &
wait
```

**If matching `not_target`/`wontfix` issue found:** `is_not_target = True`, `verdict = "Not applicable"`, `confidence = "HIGH"`
**If no match:** Continue to Step 4.

### Step 4: Check test source implementation and error message

Load the `xpu_test_knowledge` skill to understand the test file structure, registration methods, and XPU API patterns before analyzing the source code. This helps distinguish between different test registration mechanisms (e.g., `instantiate_device_type_tests` vs `XPUPatchForImport` vs `_xpu.py` wrappers) and interpret the code structure correctly.

```python
# The subagent should load xpu_test_knowledge for code structure context:
task(subagent_type="explore", load_skills=["check_not_target_feature", "xpu_test_knowledge"], ...)
```

Read the test method body from the source file and cross-reference with the `error_message` (if provided). Determine whether the test exercises CUDA-specific behavior that makes it inherently not-target for XPU.

**Read the test source:**

```bash
# Read the test class and method from the source file:
grep -n "class ${class_name}" "$PYTORCH_SRC/${test_file}"
# Then read a window around the method definition
```

**Analyze the test body for CUDA-specific patterns:**

| Pattern | Signal |
|---------|--------|
| `torch.cuda.*` calls in assertions or logic | Test exercises CUDA-specific API directly |
| `torch.backends.cuda.*` | CUDA backend-specific query |
| `torch.cuda.jiterator` | Jiterator is not supported on XPU (Issue #2918) |
| cuBLAS/cuDNN references | CUDA library-specific behavior |
| CUDA hardware assumptions (warp size, SM count, shared memory) | XPU has different hardware properties |
| CUDA-specific error strings in `assertRaises`/`pytest.raises` | Test expects CUDA-specific errors |

**Analyze the error message (if provided):**

| Error Message Signal | Meaning |
|----------------------|---------|
| `"CUDA error"`, `"cuBLAS"`, `"cuDNN"`, `"NCCL"` | CUDA-specific runtime error |
| `"is not implemented for xpu"` | XPU op is missing — enablement gap |
| `"expected CUDA"`, `"expected cuda"` | Test or API was hardcoded for CUDA |
| `"XPU backend"`, `"SYCL"`, `"Level Zero"` | XPU-specific error — not a CUDA-only problem |

**Decision rules:**

1. If the test body uses CUDA-specific APIs AND the underlying API is listed in the "Not applicable" sheet (Step 2) → `is_not_target = True`, `verdict = "Not applicable"`
2. If the test body uses CUDA-specific APIs but the API is NOT in the "Not applicable" sheet → `is_not_target = False`, `verdict = "Not not-target"` (enablement gap)
3. If the error message indicates a CUDA-only feature → cross-reference with Step 2. If not in the sheet, default to enablement gap.
4. If the test body uses device-agnostic APIs (`torch.ones`, `torch.add`, `torch.Stream()`) with CUDA parametrization → device-agnostic test, not not-target.

## Quick Reference Tables

### Not-target (Not applicable)

| Signal | Evidence Required |
|--------|------------------|
| `Operation/API` match in "Not applicable" sheet | Issue ID + Category |
| `not_target`/`wontfix` labeled issue | Issue URL verified via `gh issue view` |
| CPU-only test (`_cpu` suffix) | Test name evidence (`CPU Case`) |
| Owner-team scope explicitly non-XPU (e.g. JIT) | Owner-team label citation |
| Test body uses CUDA-specific API matched in "Not applicable" sheet | Step 2 + Step 4 evidence |
| Error message references CUDA-only feature matched in "Not applicable" sheet | Step 2 + Step 4 evidence |

## Execution

This skill is intended to be loaded and executed by a single subagent. The orchestrator delegates the entire check as one task:

```python
task(
    subagent_type="explore",
    load_skills=["check_not_target_feature"],
    description="Check if test is not-target for XPU",
    prompt="Check if <test_name> in <class_name> (<device>) is not-target for XPU. Test file: <test_file>. Error message (if any): <error_message>. Test code block: <test_code_block>. Traceback: <traceback>."
)
```

The subagent then follows the Decision Flow above, using its own tools (`bash`, `read`, `grep`, `gh`) and running independent calls in parallel. The `explore` subagent is well-suited for reading test source code and analyzing implementations.

## Example Usage

### Example 1: Not not-target (enablement gap)

```bash
# Check if test_local_stream_enter_exit in TestStreamsCUDA (cuda) is not-target
# Expected: Not not-target — no match in any authoritative source
```

```python
# Expected output:
{
    "is_not_target": False,
    "verdict": "Not not-target",
    "confidence": "HIGH",
    "evidence": [
        "Step 1: Not CPU-only — test is a CUDA test",
        "Step 2: No Operation/API match in Not applicable sheet",
        "Step 3: No matching not_target/wontfix issues found",
        "Step 4: Test body uses device-agnostic torch.Stream() API — not inherently CUDA-specific",
    ],
    "reasoning": "Test does not match any authoritative not-target source (Not applicable sheet, not_target/wontfix issues, implementation analysis). Default: Not not-target — enablement gap.",
    "suggested_reason": "To be enabled",
}
```

### Example 2: Not applicable — via "Not applicable" sheet

```bash
# Check if test_cudnn_rnn in TestFakeTensor (cuda) is not-target
# Expected: Not applicable — aten::_cudnn_rnn matches the Not applicable sheet
```

```python
# Expected output:
{
    "is_not_target": True,
    "verdict": "Not applicable",
    "confidence": "HIGH",
    "evidence": [
        "Step 1: Not CPU-only — test is a CUDA test",
        "Step 2: Match found in Not applicable sheet: 'aten::_cudnn_rnn (test_fake_tensor_xpu test_cudnn_rnn_*)'",
        "Step 4: Test body exercises aten::_cudnn_rnn (cuDNN-specific) — cross-referenced with Step 2 match",
    ],
    "reasoning": "The test exercises aten::_cudnn_rnn which is listed in the authoritative Not applicable sheet. XPU does not implement cuDNN-specific ops. Verdict: Not applicable.",
    "suggested_reason": "Not applicable",
}
```

### Example 3: Not applicable — via error message + implementation

```bash
# Check if test_stream_pointer in TestStreams (cuda) is not-target
# Error: "RuntimeError: CUDA error: invalid argument"
# Expected: Not applicable — CUDA-specific error, test body uses cuStream API
```

```python
# Expected output:
{
    "is_not_target": True,
    "verdict": "Not applicable",
    "confidence": "MEDIUM",
    "evidence": [
        "Step 1: Not CPU-only — test is a CUDA test",
        "Step 2: No Operation/API match in Not applicable sheet",
        "Step 3: No matching not_target/wontfix issues found",
        "Step 4: Error message 'CUDA error: invalid argument' indicates CUDA runtime behavior; test body uses torch.cuda.Stream which has no device-agnostic equivalent in this context",
    ],
    "reasoning": "The error message references a CUDA runtime error, and the test body exercises CUDA stream pointer semantics that have no XPU equivalent. Combined evidence supports Not applicable.",
    "suggested_reason": "Not applicable",
}
```

## Hard Constraints

- The "Not applicable" sheet is the **sole authoritative source** for CUDA-only scope decisions.
- Decorators like `@tf32_on_and_off` are CUDA-specific implementation details, not scope decisions.
- Parametrization that omits XPU (`@dtypesIfCUDA` without `@dtypesIfXPU`) is an enablement gap, not a permanent exclusion.
- When uncertain, default to `Not not-target` (`To be enabled`) rather than `Not applicable`.
- JIT clusters (`oncall:jit`, `test_jit*`, `torch.jit.*`) are `Not applicable` by owner-team scope — this is a carve-out from the general rule.
- **Never read the input Excel file directly.** All test metadata (`test_file`, `class_name`, `test_name`, `message_xpu`, etc.) is provided as task parameters by the orchestrator from the extracted `tasks.json`. Reading the Excel yourself wastes tokens and bypasses deduplication.
- **Analyze from scratch — ignore any pre-existing `Reason` or `DetailReason` in the input task data.** Prior classification results from other gates or earlier runs are irrelevant to this skill's analysis. Base your verdict solely on the test source code, error messages, and scope rules defined here. Never carry forward or reuse another gate's classification.
