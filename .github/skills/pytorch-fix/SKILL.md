---
name: pytorch-fix
description: >
  Fix a triaged PyTorch CI failure for Intel XPU. The issue body contains
  root cause analysis and proposed fix strategy — follow them.
---

# Fix PyTorch XPU CI Failure

## When to Use
When implementing a fix for a triaged issue that has root cause analysis
and proposed fix strategy in the issue body.

## Pre-flight
1. Read `.github/copilot-instructions.md` for repo context.
2. Read the issue's **Root Cause Analysis** and **Proposed Fix Strategy** sections.
3. Understand what test is failing and why before touching code.

## Fix Process

### Step 1: Reproduce
```bash
# Run the failing test(s) from the issue's Reproducer section
python -m pytest <test_file> -k <test_name> -x 2>&1 | tail -50
```
If the reproducer is missing, construct one from the failed test name.

### Step 2: Implement the Fix
Follow the **Proposed Fix Strategy** from the issue. Key principles:
- **Minimal changes** — fix only what's broken
- **XPU-specific paths** when possible (`aten/src/ATen/xpu/`, `torch/xpu/`)
- **Never skip tests** — no `@skipIfXpu`, `@skip`, `unittest.skip`
- **Never modify submodules** — no changes to `third_party/*`

### Step 3: Verify
```bash
# Run the previously failing test
python -m pytest <test_file> -k <test_name> -x

# Run related tests to check for regressions
python -m pytest <test_file> -x --timeout 120
```

### Step 4: Clean Up
```bash
# Stage only your changes (exclude third_party, submodules)
git add <your_files>
git diff --cached --stat  # verify only intended files
```

## Common Fix Patterns

### Missing XPU kernel dispatch
```cpp
// In aten/src/ATen/native/xpu/<op_name>.cpp or register in native_functions.yaml
TORCH_IMPL_FUNC(<op>_xpu) { ... }
```

### Tolerance fix
```python
# Adjust atol/rtol for XPU precision characteristics
self.assertEqual(result, expected, atol=1e-3, rtol=1e-3)
```

### Missing device propagation
```python
# Ensure output tensor is on the same device as input
output = torch.empty_like(input)  # inherits device
```

## HARD RULES
- NEVER add skip decorators. FIX the test.
- NEVER commit submodule changes (`third_party/*`).
- NEVER modify unrelated files.
- Use `git add` on specific files only.

## Output
At the end, output:
```
### Agent Summary
- **What I found:** <root cause in one sentence>
- **What I changed:** <bullet list of files>
- **Test result:** <PASS/FAIL with test command>
- **Open questions / risks:** <concerns or "None">
```
