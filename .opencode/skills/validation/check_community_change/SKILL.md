---
name: check-community-change
description: Check whether a test case is affected by an upstream community change (base function removed, renamed, or refactored in pytorch/pytorch). Verifies test case existence via pytest --collect-only when the target device is available, and falls back to AST-based source inspection otherwise. Use when determining why an XPU/CUDA test case is absent, classifying community_change vs to_be_enabled gaps, or when the base test method or device variant may have been removed/renamed/refactored upstream.
---

# check_community_change

## Purpose

Given a test file path, class name, and test method name (optionally with a device suffix), determine whether a **community change** in upstream PyTorch has affected the test case. A community change is an upstream modification (removal, rename, refactoring) that makes a workbook/generated test case name obsolete or absent.

The skill performs two sequential checks:

1. **Base-function existence**: Does the underlying test method exist in the upstream source file?
2. **Device-specific case generation**: If the base function exists, is the device-specific test case actually generated?

The device-specific check uses **`pytest --collect-only` as the primary method** — it produces exact, fully-expanded test names including all parametrizations, OpInfo expansions, and dtype variants. When the target device is not available on the system, the skill falls back to AST-based source inspection + name formula reconstruction.

If either check reveals that the test was removed, renamed, or refactored upstream, the skill returns a community-change verdict with evidence.

## Inputs

| Field | Required | Description |
|-------|----------|-------------|
| `test_file` | **Yes** | Test file path relative to `$PYTORCH_SRC` (e.g. `test/test_ops.py`, `test/dynamo/test_streams.py`) |
| `class_name` | **Yes** | Test class name (e.g. `TestFooCUDA`, `TestStreams`) |
| `test_name` | **Yes** | Test method name (e.g. `test_foo_cuda_float32`, `test_local_stream_enter_exit`) |
| `device` | No | Target device: `"cuda"` or `"xpu"`. Controls which device-specific generation is checked. Default: `"cuda"`. |
| `PYTORCH_SRC` | No | PyTorch source checkout path. Default: `$HOME/upstream/pytorch`. |

## Output

```python
{
    "community_change": bool,               # True if the test is affected by an upstream change

    "evidence": {                            # Source evidence for each check
        "base_function": {                   # Evidence from Step 1
            "base_function_found": bool,     # Whether the test method exists in the source file
            "class_found": bool,             # Whether the class exists in the source file
            "file_found": bool,              # Whether the test file exists in PYTORCH_SRC
            "file_path": str or None,        # Resolved path to the test file (or None)
            "actual_class_name": str or None,# Class name found in source (may differ from input)
            "actual_method_name": str or None,# Method name found in source (may differ from input)
            "method_line": int or None,      # Line number of the method definition
            "search_attempts": [str],        # What was searched (class variants, name variants)
        },
        "device_case": {                     # Evidence from Step 2 (only when base function exists)
            "device_available": bool,        # Whether the target device is available for --collect-only
            "verification_method": str,      # "collect_only" or "source_inspection"
            "device_case_generated": bool,   # Whether the device-specific test case IS generated
            "collect_only_output": [str],    # Full --collect-only output lines for the target class (or None)
            "similar_names_found": [str],    # Collected test names similar to expected (rename candidates)
            "generation_blocked": bool,      # True if generation is blocked by decorators/params
            "blocker": str or None,          # What blocks generation (e.g. "only_for=('xpu',)", "dtypesIfCUDA excludes dtype")
            "has_ops_decorator": bool,       # Whether @ops decorator is present
            "ops_name": str or None,         # OpInfo name if @ops decorator found
            "dtypes_for_device": [str] or None,# dtypes list from OpInfo or decorator for target device
            "instantiate_device_type_tests": str or None,  # How the class is instantiated
            "instantiate_device_type_tests_only_for": str or None,  # only_for arg if present
        },
        "git_history": {                     # Git history evidence (when base function not found)
            "commits_touching_file": [       # Recent commits that touched the test file
                {
                    "hash": str,
                    "author": str,
                    "date": str,
                    "subject": str,
                }
            ],
            "commits_removing_method": [     # Commits that removed/renamed the method
                {
                    "hash": str,
                    "author": str,
                    "date": str,
                    "subject": str,
                    "diff_summary": str,      # What changed (e.g. "Removed test_foo, renamed to test_bar")
                }
            ],
            "file_deleted_in_commit": str or None,  # Commit that deleted the whole file
        },
    },

    "classification": {
        "change_type": str,                  # "base_function_removed" | "base_function_renamed" |
                                             # "device_case_not_generated" | "device_case_renamed" |
                                             # "not_a_community_change" | "file_deleted"
        "change_scope": str,                 # "base_test_removed" | "device_removed" | "refactored" |
                                             # "renamed" | "none"
        "old_test_name": str,                # The original test name from input
        "new_test_name": str or None,        # The new test name if renamed (or None)
        "detail_reason": str,                # Concise summary for case_existence_comments
    },

    "verdict": str,                          # "Community Change" or "Not a community change"
}
```

## Preconditions

### Environment

1. PyTorch source checkout at `$PYTORCH_SRC`
2. `git` available to inspect commit history
3. `pytest` available to run `--collect-only` (requires built torch)
4. **Device availability**: The target device (CUDA/XPU) must be available for `--collect-only` to produce device-specific test classes. If the device is unavailable, the skill falls back to source inspection. Check via:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"   # CUDA
   python3 -c "import torch; print(torch.xpu.is_available())"    # XPU
   ```

### Required Tools

| Tool | Purpose |
|------|---------|
| `read` | Read test source files, class/method bodies, OpInfo definitions |
| `bash` | Run `git log`, `git show`, file lookups, `pytest --collect-only` |
| `grep` | Find class/method definitions, decorators, `instantiate_device_type_tests` calls, OpInfo entries |
| `glob` | Find test files when the path may differ from the input |

## Workflow

### Step 1: Check Base Function Existence

The base function is the actual test method defined in the upstream PyTorch source file — the method that generates the named test case after decorators, device suffixes, dtype suffixes, and OpInfo expansion are applied.

#### 1.1 Locate the test file

Resolve the file path:

```bash
test_file_path="$PYTORCH_SRC/$test_file"
ls -la "$test_file_path"
```

If the file does not exist, check git history to determine when it was removed:

```bash
cd "$PYTORCH_SRC"
git log --oneline -10 -- "$test_file"
```

**Decision:**
- File exists → continue to 1.2
- File deleted → `file_found = False`, record the deletion commit → **community change** (`file_deleted`)

#### 1.2 Locate the class

Search for the class in the test file:

```bash
grep -n "class $class_name" "$test_file_path"
```

If `class_name` ends with `CUDA`, also search for the non-device-suffixed version:

```bash
# If class_name = "TestStreamsCUDA", also search for "TestStreams"
grep -n "class ${class_name%CUDA}" "$test_file_path"
```

If the class ends with `XPU`, search for both the XPU variant and the base:

```bash
grep -n "class $class_name" "$test_file_path"
grep -n "class ${class_name%XPU}" "$test_file_path"
```

Also search for `instantiate_device_type_tests` which generates device-specific classes:

```bash
grep -n "instantiate_device_type_tests" "$test_file_path"
```

**Decision:**
- Class found directly → record actual class name, continue to 1.3
- Class not found but `instantiate_device_type_tests` exists → the class is generated; read the base class from the call, continue to 1.3
- Class not found, no `instantiate_device_type_tests` → `class_found = False`, check git for class removal

#### 1.3 Locate the method

Strip any device/dtype/OpInfo suffixes from `test_name` to find the base method:

| Generated Name | Strip Pattern | Base Method Name |
|---|---|---|
| `test_foo_cuda_float32` | Trailing `_cuda_<dtype>` | `test_foo` |
| `test_foo_xpu_float16` | Trailing `_xpu_<dtype>` | `test_foo` |
| `test_foo_cuda` | Trailing `_cuda` | `test_foo` |
| `test_bar_cuda_float32` | Trailing `_cuda_<dtype>` | `test_bar` |
| `test_op_abs_cuda_float32` | Remove `_<opname>_cuda_<dtype>` | `test_op` |
| `test_contig_large_div_xpu_float32` | Remove `_<opname>_xpu_<dtype>` | `test_contig_large_div` |

Search for the base method:

```bash
grep -n "def $base_method" "$test_file_path"
```

**Heuristic for OpInfo-based tests:** If the name pattern suggests an OpInfo test (`test_<thing>_<opname>_<device>_<dtype>`), also search for `@ops(` which indicates OpInfo parameterization:

```bash
grep -n "@ops(" "$test_file_path"
```

**Decision:**
- Method found directly → `method_found = True`, record line number, advance to Step 2
- Method not found → search git history (1.4)

#### 1.4 Check git history for method removal

```bash
cd "$PYTORCH_SRC"
git log --oneline -20 -- "$test_file"
```

For each candidate commit, inspect the diff for the method name:

```bash
git show <commit_hash> -- "$test_file" | grep -n "test_foo\|$base_method"
```

Check specifically for:
- Method renamed: a line like `-    def test_foo` and `+    def test_bar` in the same commit
- Method removed: `-    def test_foo` without a replacement
- Method refactored: signature change, parameterization change

**Decision:**
- Method was renamed → `community_change = True`, `change_type = "base_function_renamed"`, record old/new names
- Method was removed → `community_change = True`, `change_type = "base_function_removed"`, record removal commit
- Method not found in git history (never existed) → `community_change = True`, `change_type = "base_function_removed"`, `detail_reason = "Test never existed in this source version"`

### Step 2: Check Device Availability and Select Path

This step runs only if the base function EXISTS. It determines whether the device-specific test case is actually generated from the base function.

> **Key insight**: Device-specific test classes (e.g., `TestFooCUDA`) are only created at module import time if the corresponding device is available on the system. The gate is in `common_device_type.py:845`: `if torch.cuda.is_available(): test_bases.append(CUDATestBase)`. Without the device, `pytest --collect-only` will silently omit those test classes — no error, just no output. A false negative would be indistinguishable from a real removal.

#### 2.1 Check device availability

```bash
# CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# XPU
python3 -c "import torch; print(torch.xpu.is_available())"
```

Set `device_available` in the output.

**Decision:**
- **Device available** → Path A: `pytest --collect-only` (authoritative ground truth)
- **Device NOT available** → Path B: Source inspection (AST + grep + name formula)

---

### Path A: pytest --collect-only (Device Available)

When the target device is available, `--collect-only` produces the exact set of generated test names — including all OpInfo expansions, parametrizations, dtype variants, and instantiate_device_type_tests effects. No need to reconstruct the generation logic.

#### A.1 Run --collect-only

```bash
cd "$PYTORCH_SRC" && \
python3 -m pytest "$test_file" --collect-only -q 2>/dev/null
```

Capture the full output for later evidence. Filter for the target device class:

```bash
cd "$PYTORCH_SRC" && \
python3 -m pytest "$test_file" --collect-only -q 2>/dev/null | \
  grep "::${class_name}::"
```

#### A.2 Analyze collected names

Parse the filtered output for three signals:

1. **Class existence**: Does `::ClassName::` appear at all? If not, the class wasn't instantiated — check `instantiate_device_type_tests` or the device-class gate.
2. **Exact match**: Does `::ClassName::$test_name` appear? If yes, the case exists.
3. **Similar names** (rename detection): Collect all names matching the base method prefix within the class. Compare against the expected name to detect renames.

```bash
# Collect all test names matching the base method (for rename detection)
cd "$PYTORCH_SRC" && \
python3 -m pytest "$test_file" --collect-only -q 2>/dev/null | \
  grep "::${class_name}::" | grep "$base_method"
```

#### A.3 Decision matrix

| `--collect-only` evidence | `device_case_generated` | Classification |
|---|---|---|
| Class `TestFooCUDA` NOT present | `False` | `device_case_not_generated` — check `instantiate_device_type_tests` only_for |
| Class present, exact test name found | `True` | `not_a_community_change` |
| Class present, similar name found (e.g. `test_bar_cuda_float32` vs expected `test_foo_cuda_float32`) | `True` (renamed) | `device_case_renamed` — community change |
| Class present, no similar names found | `False` | `device_case_not_generated` — check if dtype gap or other blocker |

For rename detection specifically:

```python
# Compare base method against collected names
expected_name = test_name
collected_names = [...]  # from --collect-only filtered for the class
base_prefix = base_method  # e.g. "test_foo"

# Find candidates where the base method changed
candidates = [n for n in collected_names
              if n.startswith("test_") and n != expected_name
              and n.split("_")[0:2] == expected_name.split("_")[0:2]]

# Alternatively, check if any name uses a different method name + same device/dtype
# Example: expected "test_foo_cuda_float32" but "test_bar_cuda_float32" exists
```

Set `collect_only_output` to the full filtered list and `similar_names_found` to the candidate list.

#### A.4 Record evidence

```python
{
    "device_available": True,
    "verification_method": "collect_only",
    "device_case_generated": <bool>,          # From decision matrix
    "collect_only_output": [<str>],           # Full filtered --collect-only output
    "similar_names_found": [<str>],           # Rename candidates
    "generation_blocked": <bool>,             # False when using --collect-only (we know the answer)
    "blocker": None,                          # Not needed — --collect-only is decisive
    "has_ops_decorator": <bool>,              # Still populated from source for context
    "ops_name": <str or None>,
    "dtypes_for_device": [<str> or None],
    "instantiate_device_type_tests": <str or None>,
    "instantiate_device_type_tests_only_for": <str or None>,
}
```

---

### Path B: Source Inspection (Device Unavailable)

When the target device is unavailable, reconstruct test case generation via source analysis. This path covers all the mechanisms that control device-specific test creation.

#### B.1 Check instantiate_device_type_tests

Read the call site:

```bash
grep -n "instantiate_device_type_tests" "$test_file_path"
```

Determine whether the target device is included:

```python
# Device IS included
instantiate_device_type_tests(TestFoo, globals())                       # all devices → device generated
instantiate_device_type_tests(TestFoo, globals(), only_for=("cuda",))   # CUDA-only → CUDA generated
instantiate_device_type_tests(TestFoo, globals(), only_for=("cuda", "xpu"))  # both → CUDA generated

# Device is NOT included
instantiate_device_type_tests(TestFoo, globals(), only_for=("xpu",))    # XPU-only → CUDA NOT generated
instantiate_device_type_tests(TestFoo, globals(), only_for=("cpu",))    # CPU-only → CUDA NOT generated
```

If `only_for` excludes the target device, mark `generation_blocked = True`.

#### B.2 Check decorators affecting device generation

Read the decorators on the base method:

```bash
# Read ~15 lines before the method definition
sed -n '<start-15>,<start+5>p' "$test_file_path"
```

Key decorators (shown for CUDA; substitute device name for XPU):

| Decorator | Effect on target device |
|-----------|------------------------|
| `@onlyCUDA` | CUDA-only test, CUDA IS generated |
| `@onlyXPU` | XPU-only test, XPU IS generated |
| `@onlyNativeDeviceTypes` | Runs on CUDA/XPU AND CPU |
| `@skipCUDA`/`@skipIfCuda` | Test exists but SKIPPED on CUDA |
| `@dtypesIfCUDA(...)` | Only specified dtypes generate CUDA variants |
| `@dtypesIfXPU(...)` | Only specified dtypes generate XPU variants |
| `@dtypes(...)` | All specified dtypes generate device variants |
| `@ops(...)` | OpInfo-driven; generation depends on OpInfo dtypes for device |
| `@parametrize("device", ["cpu"])` | Device parametrization WITHOUT target → NOT generated |

**Decision:**
- Explicit device exclusion → not generated → by design (not a community change)
- Device decorated but dtype-restricted → generated for subset of dtypes → check if target dtype is included

#### B.3 Check OpInfo parameterization

If the method uses `@ops(<op_name>)`, locate the OpInfo definition:

```bash
grep -n "OpInfo('<op_name>'\|BinaryUfuncInfo('<op_name>'\|UnaryUfuncInfo('<op_name>')" \
  $PYTORCH_SRC/torch/testing/_internal/common_methods_invocations.py
```

Also check in modularized definitions:

```bash
grep -rn "def.*'<op_name>'\|'<op_name>'" \
  $PYTORCH_SRC/torch/testing/_internal/opinfo/definitions/ --include="*.py"
```

Extract device-relevant fields:

```python
{
    "dtypes": ...,               # Base dtypes (applied to CPU)
    "dtypesIfCUDA": ...,         # CUDA-specific dtypes (override for CUDA)
    "dtypesIfXPU": ...,          # XPU-specific dtypes (override for XPU)
    "skips": ...,                # SkipInfo entries including device_type='cuda'/'xpu'
}
```

**Decision:**
- If `dtypesIf<DEVICE>` is set and the target dtype is NOT in the list:
  `device_case_generated = False`, `generation_blocked = True`,
  `blocker = "dtypesIf<DEVICE> excludes <dtype>"`
  This is NOT a community change — it's a device dtype gap.
- If `dtypesIf<DEVICE>` is set and the target dtype IS in the list:
  `device_case_generated = True`
- If `dtypesIf<DEVICE>` is not set and base `dtypes` includes the target dtype:
  `device_case_generated = True`
- If OpInfo has `skips` with `device_type='<device>'`:
  Variant IS generated but runtime-skipped → `device_case_generated = True`

#### B.4 Reconstruct expected test name

Use the deterministic name formula from `common_device_type.py` to verify that the expected name matches what the generation logic would produce:

```python
# For simple device-type tests (instantiate_device_type_tests):
# Generated name format: {base_method}_{device}_{dtype}
# e.g. test_foo_cuda_float32

# For OpInfo tests:
# Generated name format: test_{opinfo_category}_{op_name}_{device}_{dtype}
# where OpInfo.formatted_name = full_name.replace(".", "_")

# For parametrized tests:
# Each @parametrize argument is appended to the name
```

Strip the expected test name and reconstruct it from the source components:

1. Start with `base_method` (from Step 1.3)
2. Append `_<device>` (e.g. `_cuda`, `_xpu`)
3. Append `_<dtype>` if dtype is involved
4. For OpInfo: the name includes the OpInfo `formatted_name` before the device suffix
5. For parametrize: each parametrized value is appended as `_<value>`

If the reconstructed name differs from the input `test_name`, the test may have been renamed.

#### B.5 Record evidence

```python
{
    "device_available": False,
    "verification_method": "source_inspection",
    "device_case_generated": <bool>,          # From source analysis
    "collect_only_output": None,              # Not available
    "similar_names_found": None,              # Not available
    "generation_blocked": <bool>,             # True if blocked by decorators/params
    "blocker": <str or None>,                 # E.g. "only_for=('xpu',)", "dtypesIfCUDA excludes float32"
    "has_ops_decorator": <bool>,
    "ops_name": <str or None>,
    "dtypes_for_device": [<str> or None],
    "instantiate_device_type_tests": <str or None>,
    "instantiate_device_type_tests_only_for": <str or None>,
}
```

#### B.6 Synthesize device-case verdict (source inspection)

| Scenario | `device_case_generated` | Community Change? |
|----------|-------------------------|-------------------|
| Base function exists, device excluded by `only_for` | `False` (by design) | No — other-device-only parametrization |
| Base function exists, dtype excluded by `dtypesIf<DEVICE>` | `False` (dtype gap) | No — device dtype gap |
| Base function exists, all generation conditions satisfied | `True` | No — test should work |
| Base function exists, generation should happen but name formula mismatch | `False` | Yes — investigate (`device_case_not_generated`) |
| Base function was renamed, and device variant uses new name | `True` (new name) | Yes — old name is obsolete (`device_case_renamed`) |

### Step 3: Synthesize Community Change Verdict

Combine findings from Steps 1 and 2 into the final classification:

| Evidence | `change_type` | `change_scope` | `community_change` |
|----------|---------------|-----------------|--------------------|
| File deleted upstream | `file_deleted` | `base_test_removed` | `True` |
| Base method removed | `base_function_removed` | `base_test_removed` | `True` |
| Base method renamed (old→new) | `base_function_renamed` | `renamed` | `True` |
| Base method refactored (param change, class merge) | `base_function_renamed` | `refactored` | `True` |
| Base function exists, device excluded by `only_for` | `device_case_not_generated` | `none` | `False` |
| Base function exists, device dtype gap | `device_case_not_generated` | `none` | `False` |
| Base function exists, device variant collected | `not_a_community_change` | `none` | `False` |
| Base function exists, device variant renamed | `device_case_renamed` | `renamed` | `True` |

### Step 4: Generate Evidence-Based Detail Reason

Format the `detail_reason` string for the `case_existence_comments` column:

| Scenario | `detail_reason` |
|----------|-----------------|
| Base method removed | `"Base test removed: $base_method in $test_file (commit <hash>)"` |
| Base method renamed | `"Base test renamed: $old_name → $new_name (commit <hash>)"` |
| File deleted | `"Test file deleted: $test_file (commit <hash>)"` |
| Refactored | `"Base test refactored: $base_method ($summary, commit <hash>)"` |
| Device excluded by `only_for` | `"$device variant not generated: instantiate_device_type_tests only_for=$only_for"` |
| Device dtype gap | `"$device variant not generated: $op dtypesIf$DEVICE excludes $dtype"` |
| Device variant renamed (--collect-only) | `"$device variant renamed: $old_name → $new_name (collected: $similar_names)"` |
| Not a community change | `"Base function exists, $device variant is generated — no community change"` |

## Execution

This skill is intended to be loaded and executed by a subagent. The orchestrator delegates the entire check as one task:

```python
task(
    subagent_type="explore",
    load_skills=["check_community_change"],
    description="Check community change for test case",
    prompt="Check community change for <test_name> in <class_name> (device=<device>). "
           "Test file: <test_file>. PYTORCH_SRC=<path>. "
           "First check if CUDA is available. If yes, use --collect-only (Path A). "
           "If not, use source inspection (Path B)."
)
```

The subagent then follows the Workflow above, using its own tools (`bash`, `read`, `grep`, `glob`) and running independent calls in parallel. The subagent MUST check device availability first (Step 2.1) before deciding which path to follow — never assume the device is available.

## Constraints

1. **Base-function-first**: Always check base function existence before concluding any other absence reason. If the base function is absent, classify as Community Change regardless of device wrapper status.
2. **Device availability gate first**: Before running `--collect-only`, check whether the target device is available. If the device is unavailable, `--collect-only` will silently omit device-specific test classes — do NOT use it in that case. Fall back to source inspection.
3. **`--collect-only` is authoritative (Path A)**: When the target device is available, `--collect-only` output is the single source of truth for whether a test case exists. Source-level analysis (OpInfo dtypes, decorators, parametrization) is NOT needed — `--collect-only` accounts for all of them at runtime.
4. **Source-inspection fallback must be thorough (Path B)**: When the device is unavailable, ensure the fallback covers all generation paths: `instantiate_device_type_tests` only_for, decorators (dtypesIfDEVICE, skip, only), OpInfo dtype lists, and parametrize. Missing any path can produce a false community-change classification.
5. **Git history is authoritative**: For removed/renamed tests, `git log` and `git show` evidence takes precedence over any other signal.
6. **Name-suffix stripping**: Always strip trailing device/dtype/OpInfo suffixes from `test_name` before searching for the base method. Incorrect suffix stripping is the most common error.
7. **Device-only parametrization is not a community change**: If `instantiate_device_type_tests(only_for=("xpu",))` explicitly excludes the target device, that is by design — do not flag as community change.
8. **Device dtype gaps are not community changes**: If `dtypesIfCUDA` excludes the target dtype, that is a CUDA dtype gap, not a community change.
9. **Do not conflate "device case absent" with "community change"**: A missing device case may be a dtype gap or parametrization exclusion, not an upstream removal.

## Version

- v2.0.0 - 2026-06-09 - Bifurcated approach: --collect-only (Path A) when device available, source inspection (Path B) fallback.

## See Also

- `extract_base_function` — Provides base-function metadata from a broader perspective (used as reference for this skill's design)
- `check_not_target_feature` — Determines if a test is CUDA-only / not applicable for XPU
