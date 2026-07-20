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
- `PYTORCH_SRC` — the `pytorch_folder` the calling agent already prepared; used
  for the Step 1.5 existence pre-check and the Step 5 source grep. Use it as
  given. Do NOT set up or activate any environment (no `setup_env.sh`).

## Output Format
Return this JSON object:
```python
{
    "is_not_target": bool,
    "verdict": "Not applicable" | "Not not-target" | "CPU Case",
    "evidence": [str],      # Specific matches. If a not_target/wontfix/skipped issue exists, list its number FIRST, e.g., "intel/torch-xpu-ops#3127 (CLOSED, not_target)", then the API/sheet/code citation, e.g., "Matched 'aten::_cudnn_rnn' in Not Applicable sheet". The Step 6 backfill attaches this link deterministically.
    "reasoning": str        # Brief summary of the decision
}
```

## ⚠️ Decorator Trap — Read Before Any Analysis

The following patterns look like "CUDA-only" but are **NOT** evidence for `is_not_target = True`.
They mean the test has not been enabled for XPU yet — which is an **enablement gap** (`To be enabled`), not an out-of-scope decision.

**NEVER use these alone as grounds for `is_not_target = True`:**

| Pattern | Correct interpretation |
|---|---|
| `@onlyCUDA` | Test not yet enabled for XPU → enablement gap |
| `@requires_cuda_and_triton` | Test not yet enabled for XPU → enablement gap |
| `@requires_cuda` | Test not yet enabled for XPU → enablement gap |
| `@skipUnless(TEST_CUDA, ...)` on a class or method | Test not yet enabled for XPU → enablement gap |
| `@skipIf(not TEST_CUDA, ...)` on a class or method | Test not yet enabled for XPU → enablement gap |
| `@pytest.mark.skipif(not HAS_CUDA_AND_TRITON, ...)` | Test not yet enabled for XPU → enablement gap |
| `if HAS_CUDA_AND_TRITON: class Foo(...)` | Class not yet created for XPU → enablement gap |
| `if self.device != "cuda": raise SkipTest(...)` | Test not yet parameterized for XPU → enablement gap |
| `if self.device not in ("cpu", "cuda"): raise SkipTest(...)` | Test not yet parameterized for XPU → enablement gap |
| `instantiate_device_type_tests(..., only_for=("cuda",))` | XPU instantiation not yet added → enablement gap |
| `raise SkipTest("requires CUDA")` | Test not yet enabled for XPU → enablement gap |
| `raise SkipTest("requires CUDA/HIP")` | Test not yet enabled for XPU → enablement gap |

**The only valid grounds for `is_not_target = True` are (in order of precedence):**
1. A match in the Not-Applicable sheet (Step 3)
2. A CLOSED `not_target` or `wontfix` issue in `intel/torch-xpu-ops` (Step 4)
3. The test body calls a **strictly CUDA-hardware-tied API** with no XPU equivalent (Step 5) — defined strictly below

**A missing test is never grounds for `is_not_target = True`.** If the test file
or function is absent from the source (removed/renamed/refactored upstream),
return `is_not_target = False` and let Gate 2 (`check-community-change`) handle
it. This is enforced by the mandatory Step 1.5 existence pre-check, which runs
before any sheet/issue/code analysis.

## Deep Analysis Workflow

### 1. Mandatory Input Scrubbing
- **Ignore** any pre-existing `Reason` or `DetailReason` from the task input. Do not carry them forward.
- **Never read the Excel file directly.**
- **Export `PYTORCH_SRC`** before any command so the Step 1.5 existence
  pre-check and Step 5 source grep resolve correctly (an unset `$PYTORCH_SRC`
  makes `$PYTORCH_SRC/$test_file` expand from filesystem root and silently match
  nothing). Do NOT set up or activate any environment:
  ```bash
  export PYTORCH_SRC="<pytorch_folder the caller provided>"
  ```

### 1.5 Mandatory Existence Pre-Check (run FIRST, before Steps 2-6 — hard gate)

Before any sheet lookup, issue search, or code inspection, verify the test file
and base method actually exist in `PYTORCH_SRC`. A missing file or method means
the test was removed/renamed/refactored upstream — a **community change**, never
a not-target feature. This gate makes it structurally impossible to reach an
`is_not_target = True` verdict for an absent test, regardless of what the sheet
(Step 3) or issue search (Step 4) might otherwise match by name.

Derive `base_method` from `test_name` by stripping the trailing device suffix
(`_cuda`/`_xpu`) and any dtype suffix (`_float32`, `_bfloat16`, ...).

```bash
FILE="$PYTORCH_SRC/$test_file"
if [ ! -f "$FILE" ]; then
    echo "MISSING_FILE"
elif ! grep -Eq "def[[:space:]]+${base_method}\b" "$FILE"; then
    echo "MISSING_METHOD"
else
    echo "PRESENT"
fi
```

- **`MISSING_FILE`**: The test file does not exist. Return immediately:
  `is_not_target = False`, `verdict = "Not not-target"`,
  `reasoning = "test file <test_file> not found in source; defer to community-change gate"`,
  `evidence = ["test file <test_file> not found in PYTORCH_SRC"]`.
  **Do NOT run Steps 2-6.**
- **`MISSING_METHOD`**: The file exists but has no literal `def <base_method>`.
  This can be an upstream removal/rename OR a generated test
  (OpInfo / `instantiate_device_type_tests`) — either way it is not a not-target
  decision here. Return immediately: `is_not_target = False`,
  `verdict = "Not not-target"`,
  `reasoning = "base method <base_method> not found in <test_file>; defer to community-change gate"`,
  `evidence = ["no 'def <base_method>' in <test_file>"]`.
  **Do NOT run Steps 2-6.**
- **`PRESENT`**: Both file and method exist → proceed to Step 2.

This gate is the authoritative implementation of the Missing-Test Guard: it runs
BEFORE Step 3/Step 4, so a name-glob or issue-title match can never override a
genuinely missing test.

### 2. CPU-Only Fast Path
If the test name ends with `_cpu`, contains `_cpu_`, or the skip message explicitly says "requires GPU" but targets CPU:
- `is_not_target = False`, `verdict = "CPU Case"`

### 3. "Not Applicable" Sheet Check (Authoritative)
The sheet at `https://github.com/daisyden/ai_for_validation/blob/main/opencode/issue_triage/result/torch_xpu_ops_issues.xlsx` is the absolute source of truth.

Run the bundled script to fetch the current JSON export of this sheet:
```bash
python3 .opencode/skills/validation/scripts/list_not_applicable.py --json
```

**Match Extraction**:
- **Test Identity**: Match `test_name` against globs in the sheet (e.g., `test_fake_tensor_xpu test_cudnn_rnn_*`).
- **Code/Traceback APIs**: Look for `aten::*`, `torch.*` in the test body or traceback, and see if they exist in the `Operation/API` column (e.g., `aten::_cudnn_rnn`).
- **Error Keywords**: Match explicit strings in the error message (e.g., `"cuDNN"`, `"jiterator"`) against the sheet descriptions.

If a match is found:
- `is_not_target = True`, `verdict = "Not applicable"`. Evidence must cite the exact `Operation/API` entry. Then run Step 6 to attach a `not_target`/`wontfix` issue number if one exists.

**If no match → proceed to Step 4. Do NOT skip to Step 5.**

### 4. Known "Not Target" Issues
Run parallel searches to check if the specific test or API was already closed as `not_target`/`wontfix` in `intel/torch-xpu-ops`:
```bash
gh search issues "<test_name_no_suffix> not_target is:issue" --repo=intel/torch-xpu-ops --limit=3 &
gh search issues "<test_name_no_suffix> wontfix is:issue" --repo=intel/torch-xpu-ops --limit=3 &
wait
```
If matched and verified via `gh issue view` (state=CLOSED, label=`not_target`|`wontfix`):
- `is_not_target = True`, `verdict = "Not applicable"`. Evidence must list the issue number and URL, e.g., `intel/torch-xpu-ops#3127 (CLOSED, not_target) https://github.com/intel/torch-xpu-ops/issues/3127`.

**If no match → proceed to Step 5. Do NOT declare `is_not_target = True` yet.**

### 5. Implementation Analysis (Last-Resort Fallback)

**GATE CHECK — before inspecting any code, confirm both:**
- [ ] Step 3 found NO sheet match
- [ ] Step 4 found NO closed `not_target`/`wontfix` issue

If both are confirmed, inspect the test body:
```bash
grep -A 20 -n "def ${test_name_no_suffix}" "$PYTORCH_SRC/${test_file}"
```

**MISSING-TEST GUARD (mandatory; normally already handled by Step 1.5):** The
Step 1.5 existence pre-check should have already returned `is_not_target = False`
for any missing file/method before reaching this step. As a backstop, if the test
file does not exist, or the test function is not found in the source at all (the
grep above returns nothing), this is **NOT** a not_target signal. A
missing/absent/removed/renamed test means the test was removed or refactored
upstream — that is a **community change**, handled by Gate 2
(`check-community-change`), not an out-of-scope decision. In this case you MUST
return `is_not_target = False` (reasoning: "test not found in source; defer to
community-change gate") and let the cascade fall through to Gate 2. Never set
`is_not_target = True` on the
grounds that a test is "stale", "missing", "absent", "not present in current
source", or "mismatched identifier".

**`is_not_target = True` ONLY if the test body contains one of these strictly CUDA-hardware-tied patterns with no XPU equivalent:**
- Direct calls to CUDA-proprietary APIs: `torch.cuda.jiterator`, `aten::_cudnn_rnn`, `aten::_cudnn_*`, `torch._C._broadcast_coalesced`, `torch.cuda._record_memory_history`
- Assertions on CUDA hardware internals that have no XPU counterpart: SM count, warp size, PTX instructions
- Tests that validate CUDA-specific error messages (e.g., integer bmm CUDA error text) where the behavior differs by design
- JIT owner-team scope (`torch.jit.*`, `oncall:jit`) — implicitly `Not applicable`

**`is_not_target = False` for ALL of the following (even if found in the test):**
- Any decorator from the Decorator Trap table above (`@onlyCUDA`, `@requires_cuda_and_triton`, `@skipUnless(TEST_CUDA)`, etc.)
- `if self.device != "cuda": raise SkipTest(...)` — this is a parametrization gap
- `if HAS_CUDA_AND_TRITON: class Foo(...)` — this is a missing XPU class
- `instantiate_device_type_tests(..., only_for=("cuda",))` — missing XPU instantiation
- Standard APIs (`torch.add`, `torch.mm`, `torch.Stream`) with a CUDA suffix — device-agnostic parametrization gap
- Missing `@dtypesIfXPU` or missing XPU dtype coverage — parametrization gap
- `"is not implemented for xpu"` — missing op (enablement gap)
- `skipIfXpu` with `"FIXME"` or `"doesn't currently work"` messages — explicit enablement gap
- Any test that uses generic GPU operations (matmul, activations, element-wise ops) — enablement gap

If the test is device-agnostic parametrization or just missing an XPU implementation → `is_not_target = False`.

### 6. Attach Tracking Issue Number (deterministic backfill; mandatory for every `Not applicable` verdict)

Whenever the verdict is `Not applicable`, the `evidence` MUST carry the
corresponding `intel/torch-xpu-ops` tracking issue link when one exists — even
when the verdict was reached via the sheet (Step 3) or implementation analysis
(Step 5), which do not inherently produce an issue (e.g. XNNPACK skips are
tracked by `intel/torch-xpu-ops#4179`, labeled `not_target`+`skipped`, which
lists the exact test names in its body). This backfill attaches the link as
**supporting evidence without changing the verdict** — it is a cheap, bounded,
deterministic lookup, NOT a full known-issue search.

1. If Step 4 already matched a CLOSED `not_target`/`wontfix` issue, that number is
   the evidence — done. Do not run the script.
2. If `evidence` already contains a `github.com/.../issues/NNN` URL — done. No lookup.
3. Otherwise, run the bounded deterministic lookup script once:
   ```bash
   python3 .opencode/skills/validation/scripts/attach_not_target_evidence.py \
       --name-xpu "<test_name>" --classname-xpu "<class_name>"
   ```
   Pass the XPU test identifiers: `<test_name>` and `<class_name>` are this
   invocation's inputs (the XPU-side name/class the caller supplied). The script
   dumps closed `not_target` and open `skipped` issues from `intel/torch-xpu-ops`
   (with bodies) and locally string-matches the test name and class name against
   each body (the same deterministic anchor `check-known-issue` uses). It prints a
   JSON object with `matched`, `issue_number`, `url`, `title`, `state`, `labels`.
4. On a literal match (`matched == true`), prepend the issue to `evidence`
   (format `intel/torch-xpu-ops#<number> (<state>, <labels>) <url>`). The verdict
   stays `Not applicable`.
5. If no match, keep the sheet/code citation as the evidence and append a note
   that no tracking issue was found. Never fabricate an issue number.


## Strict Constraints (ZERO TOLERANCE)

1. **Default to Enablement**: If Steps 3, 4, and 5 all fail to produce a qualifying match, `is_not_target = False`. Skipping when CUDA is absent means the test **has not been enabled for XPU yet** — that is an enablement gap, not an out-of-scope decision. CUDA decorators (`@onlyCUDA`, `@requires_cuda_and_triton`, `@skipUnless(TEST_CUDA)`, etc.) are **never** sufficient alone for `is_not_target = True`.
2. **Missing Ops are NOT "Not Applicable"**: An error like `"is not implemented for xpu"` means it is missing (enablement gap), not out of scope.
3. **Parametrization gaps are NOT "Not Applicable"**: Missing `@dtypesIfXPU` when `@dtypesIfCUDA` exists is a gap, not a scope decision. A test class defined only inside `if HAS_CUDA_AND_TRITON:` is a missing XPU instantiation, not a scope decision.
4. **Tool Restriction**: Use `bash` (with `gh` and python scripts, including `attach_not_target_evidence.py`), `read`, `grep`. No web tools. `gh search issues` must use `is:issue` to filter PRs out.
5. **No Blind Copies**: Do not copy input classification columns. Evaluate from scratch.
6. **Prefer Issue Numbers as Evidence**: When the verdict is `Not applicable` and a corresponding CLOSED `not_target`/`wontfix` (or open `skipped`) issue exists, its number MUST appear in `evidence` (format `intel/torch-xpu-ops#NNNN`). Run the Step 6 deterministic backfill (`attach_not_target_evidence.py`) to find it. Only fall back to a sheet `Operation/API` entry or a `file:line` code citation when the backfill genuinely returns no match. Never invent an issue number.
7. **Steps are sequential and exhaustive**: You MUST run Step 1.5 (existence pre-check) FIRST; if it returns `MISSING_FILE`/`MISSING_METHOD`, return `is_not_target = False` immediately and run nothing else. Otherwise you MUST run Step 3 (sheet check) and Step 4 (issue search) before running Step 5. You MUST NOT declare `is_not_target = True` from Step 5 based on decorators alone. The gate check at the top of Step 5 is mandatory.
8. **Missing tests are NOT "Not Applicable"**: If the test file or test function cannot be found in the source (`$PYTORCH_SRC`), that is evidence of an upstream removal/rename/refactor — a **community change** to be resolved by Gate 2, NOT grounds for `is_not_target = True`. A missing, absent, stale, or mismatched test identifier MUST yield `is_not_target = False`. This is enforced deterministically by the mandatory Step 1.5 existence pre-check, which runs before the sheet lookup and issue search so a name-glob or issue-title match can never override a genuinely missing test.
