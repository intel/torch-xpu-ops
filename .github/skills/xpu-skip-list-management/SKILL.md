---
name: xpu-skip-list-management
description: >
  Add, remove, or manage test skip lists for XPU platforms in the torch-xpu-ops
  repository. Use when skipping failing tests, removing skips for fixed operators,
  managing platform-specific skips (Arc, MTL, BMG, LNL, PVC), or editing
  skip_list_*.py files.
---

# XPU Skip List Management

This Skill guides you through correctly managing the test skip list system across platforms.

Read `.github/copilot-instructions.md` for full repo context before starting.

---

## Overview: Skip list hierarchy

Skip lists control which tests are skipped during CI runs. They are organized in a hierarchy:

```
test/xpu/
├── skip_list_common.py       # All platforms — default skip list
├── skip_list_arc.py          # Intel Arc (DG2) specific
├── skip_list_mtl.py          # Intel Meteor Lake specific
├── skip_list_bmg.py          # Intel Battlemage specific (if exists)
├── skip_list_lnl.py          # Intel Lunar Lake specific (if exists)
├── skip_list_dist.py         # Distributed tests
├── skip_list_win.py          # Windows (all platforms)
├── skip_list_win_arc.py      # Windows + Arc
├── skip_list_win_mtl.py      # Windows + MTL
├── skip_list_win_bmg.py      # Windows + BMG
├── skip_list_win_lnl.py      # Windows + LNL
└── windows_skip_dict.py      # Windows skip dictionary (merged at runtime)
```

**Hierarchy rule**: `skip_list_common.py` applies to ALL platforms. Platform-specific files add additional skips on top of common.

---

## Step 1: Determine the right skip list

| Situation | File to edit |
|-----------|-------------|
| Test fails on ALL XPU platforms | `skip_list_common.py` |
| Test fails only on Arc (DG2) | `skip_list_arc.py` |
| Test fails only on Meteor Lake | `skip_list_mtl.py` |
| Test fails only on Battlemage | `skip_list_bmg.py` |
| Test fails only on Lunar Lake | `skip_list_lnl.py` |
| Test fails on all Windows | `skip_list_win.py` |
| Test fails on Windows + specific platform | `skip_list_win_<platform>.py` |
| Distributed test failure | `skip_list_dist.py` |

**When in doubt**, add to `skip_list_common.py` and narrow later.

---

## Step 2: Understand the format

Each skip list file defines a `skip_dict` dictionary:

```python
skip_dict = {
    "test_file_xpu.py": (
        # Comment explaining why this test is skipped
        "test_case_name_xpu",
        # Another comment
        "test_another_case_xpu",
    ),
    "test_other_file_xpu.py": None,  # Skip entire file
}
```

### Key format rules

1. **Keys** are test file names (relative to `test/xpu/`), including subdirectory paths:
   ```python
   "test_nn_xpu.py": (...)
   "nn/test_convolution_xpu.py": (...)
   "quantization/core/test_quantized_op_xpu.py": (...)
   ```

2. **Values** are either:
   - `None` — skip the entire test file
   - A tuple of test case name strings — skip only those specific tests
   - A tuple with substring patterns — any test containing the substring is skipped:
     ```python
     "test_binary_ufuncs_xpu.py": ("_jiterator_",),  # skips all jiterator tests
     ```

3. **Comments** explaining the skip reason should precede each entry:
   ```python
   # AssertionError: Tensor-likes are not close!
   # RuntimeError: value cannot be converted to type int without overflow
   "test_add_scalar_relu_xpu",
   ```

---

## Step 3: Add a skip

### Adding a single test skip

```python
skip_dict = {
    # ... existing entries ...
    "test_ops_xpu.py": (
        # ... existing skips ...
        # <issue link or error description>
        "test_new_failing_case_xpu",
    ),
}
```

### Adding a skip to an existing file entry

If the file key already exists, add to its tuple:

```python
"test_nn_xpu.py": (
    # existing skips...
    "test_existing_skip_xpu",
    # New: <reason>
    "test_new_skip_xpu",
),
```

### Skipping an entire test file

```python
"nn/test_new_module_xpu.py": None,
```

---

## Step 4: Remove a skip

When an op is fixed or a test starts passing:

1. **Remove the test name** from the skip tuple
2. **Remove the associated comment**
3. **If the tuple becomes empty**, remove the entire file entry
4. **Verify** the test actually passes before removing:
   ```bash
   python test/xpu/run_test_with_skip.py test/xpu/test_file_xpu.py -k "test_case_name"
   ```

---

## Step 5: Understand the driver scripts

Skip lists are consumed by platform-specific test runners:

| Runner script | Skip lists used |
|---------------|----------------|
| `run_test_with_skip.py` | `skip_list_common.py` |
| `run_test_with_skip_arc.py` | `skip_list_common.py` + `skip_list_arc.py` |
| `run_test_with_skip_mtl.py` | `skip_list_common.py` + `skip_list_mtl.py` |
| `run_test_with_skip_bmg.py` | `skip_list_common.py` + `skip_list_bmg.py` |
| `run_test_with_skip_lnl.py` | `skip_list_common.py` + `skip_list_lnl.py` |

On Windows, `windows_skip_dict.py` and `skip_list_win*.py` are merged with the Linux skip lists at runtime.

---

## Step 6: Best practices

### Always include a reason

```python
# Good:
# RuntimeError: int overflow in index calculation — tracked in #1234
"test_large_tensor_indexing_xpu",

# Bad (no reason):
"test_large_tensor_indexing_xpu",
```

### Prefer specific test names over substring patterns

```python
# Good — skips exactly one test:
"test_specific_failure_case_xpu_float32",

# Use with caution — skips all matching tests:
"_jiterator_",
```

### Keep skips temporary

Skips should be treated as tech debt. When adding a skip:
- Reference the tracking issue (e.g., `# tracked in #1234`)
- Mark with a TODO if the fix is known but not yet applied

### Verify before removing

Always run the specific test before removing a skip:
```bash
# Run just the test that was skipped
python test/xpu/run_test_with_skip.py test/xpu/test_file_xpu.py -k "test_case_name"

# Run the full file to check for regressions
python test/xpu/run_test_with_skip.py test/xpu/test_file_xpu.py
```

---

## Common mistakes

1. **Wrong skip list file** — adding a platform-specific skip to `skip_list_common.py` (unnecessarily skips on all platforms)
2. **Missing comma in tuple** — a single-element tuple needs a trailing comma:
   ```python
   # Wrong:
   "test_file.py": ("test_case")      # This is a string, not a tuple!
   # Right:
   "test_file.py": ("test_case",)     # Trailing comma makes it a tuple
   ```
3. **No skip reason** — makes it impossible to know when the skip can be removed
4. **Stale skips** — test was fixed but skip was never removed
5. **Wrong test name** — test name doesn't match exactly (no silent failure — the skip just has no effect)

---

## Checklist

- [ ] Correct skip list file chosen (common vs platform-specific)
- [ ] Comment explaining the skip reason is present
- [ ] Trailing comma on single-element tuples
- [ ] Test name matches exactly (verify by running the test)
- [ ] Issue reference included where applicable
- [ ] For skip removal: verified the test passes before removing
