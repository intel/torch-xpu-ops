# Fix Lintrunner Issues - Examples

## Real-World Examples from PR #3383

These examples demonstrate the deep semantic analysis required for fixing lint issues in torch-xpu-ops.

## Example 1: E731 Lambda Warnings (Triton Grid Functions)

### Error Context
```bash
lintrunner -a test/xpu/dynamo/test_aot_autograd_cache_xpu.py
# Results in E731 warnings:
# Line 1039: grid = lambda meta: (
#             triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
#         )
```

### WRONG Analysis Approach (Pattern Matching)
```
User might suggest: "Just replace lambda with def"
Attempted fix:
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
# WARNING: This breaks the code because context is lost
```

### RIGHT Analysis Approach (Semantic Deep Dive)

```python
# Step 1: Read surrounding context
# Lines 1035-1043:
kernel_fn = inner_kernel  # Direct assignment from global
wrapped_kernel = identity_wrapper(kernel_fn)  # Wrapper call
grid = lambda meta: (
    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
)  # noqa: E731
capture_triton(wrapped_kernel)[grid](y, n_elements, BLOCK_SIZE=256)
return y
```

**Step 2: Understand semantic purpose**
- `capture_triton` requires callable `grid`
- Lambda provides inline callable with captured `n_elements`
- Context: Function inside squeeze_inner_kernel for triton op

**Step 3: Decision**
- This lambda CANNOT be replaced with regular def
- It's called immediately: `capture_triton(kernel)[grid](...)`
- Def would require module-level scope changes
- Required pattern for triton kernel grid scheduling

**Step 4: Apply fix correctly**
```python
# OLD:
grid = lambda meta: (
    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
)  # noqa: E731

# CORRECTED: If noqa is on WRONG line
# WRONG:
grid = lambda meta: (
    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
)  # noqa: E731
# This is on closing parenthesis, not same line as lambda

# RIGHT: noqa on SAME line as lambda definition
grid = lambda meta: (  # noqa: E731
    triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
)
```

### Verification Command
```bash
lintrunner -a test/xpu/dynamo/test_aot_autograd_cache_xpu.py 2>&1 | grep E731
# Should show NO errors after fix
```

## Example 2: B950 Line-Too-Long in Expected Output

### Error Context
```bash
# test/xpu/dynamo/test_higher_order_ops_xpu.py lines 750-755
c: "i64[u0, 1]" = l_x_.nonzero()
sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(c, 0)
ge: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
>>> 752  |_assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion..."); ge = _assert_scalar_default = None
wrap_body_1 = self.wrap_body_1
>>> 755  |wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, l_x_, sym_size_int_1, c); ...
```

### WRONG Analysis Approach
```
User might suggest: "Auto-break all lines over 120 chars"
# Results in broken expected output:
assertExpectedRaisesInline(
    foo(),
    """
wr
ong
  bro
ken
line
"""
)
```

### RIGHT Analysis Approach (Semantic Deep Dive)

```python
# Step 1: Identify context
# This is inside assertExpectedRaisesInline() call
# The content is EXPECTED OUTPUT from torch.compile
# It's NOT user code - it's captured Dynamo output

# Step 2: Apply B950 noqa rule correctly
# Per AGENTS.md: put noqa on SAME LINE as TERMINATING TRIPLE QUOTE
self.assertExpectedInline(
    actual,
    """\
verbose Irwin debug traceback:
...
>>> 752  |...
wrap = torch.ops.higher_order.wrap(wrap_body_1, s77, l_x_, ...); ...
""",  # noqa: B950  <-- CORRECT LOCATION
)
```

## Example 3: META_NO_CREATE_UNBACKED Errors (PR intel/torch-xpu-ops#3383)

### Error Context
```bash
# test/xpu/dynamo/test_misc_xpu.py lines 11677-11679, 11831-11832
main.create_unbacked_symint()   # noqa: META_NO_CREATE_UNBACKED
main.create_unbacked_symfloat() # noqa: META_NO_CREATE_UNBACKED
main.create_unbacked_symbool()  # noqa: META_NO_CREATE_UNBACKED
# preci-lint-check still reports 6 errors despite the noqa comments.
```

### Why the noqa comments don't help

Inspecting `.lintrunner.toml` in torch-xpu-ops:

```toml
[[linter]]
code = 'META_NO_CREATE_UNBACKED'
include_patterns = ["**/*.py"]     # ← applies repo-wide, unlike pytorch
command = [
    'python3', 'tools/linter/adapters/grep_linter.py',
    '--pattern=create_unbacked',
    '--linter-name=META_NO_CREATE_UNBACKED',
    ...
]
```

`grep_linter.py` without `--allowlist-pattern` is a raw grep. It does **not** understand `# noqa: <CODE>`. So the noqa comments are dead code; they neither suppress the error nor document anything useful.

Compare with upstream `pytorch/pytorch/.lintrunner.toml`, where this rule is scoped to `include_patterns = ["torch/_meta_registrations.py"]`. Upstream `test/dynamo/test_misc.py` calls `create_unbacked_*` freely with no noqa — because the rule never scans it.

### Correct fix (applied in PR #3383)

1. Add the ported test file to `exclude_patterns` for that one rule:

   ```toml
   [[linter]]
   code = 'META_NO_CREATE_UNBACKED'
   include_patterns = ["**/*.py"]
   exclude_patterns = [
     # Port of upstream test/dynamo/test_misc.py which legitimately calls
     # ShapeEnv.create_unbacked_* to test ShapeEnv equality.
     "test/xpu/dynamo/test_misc_xpu.py",
   ]
   ```

2. Delete the dead `# noqa: META_NO_CREATE_UNBACKED` comments from the test file so it matches upstream verbatim.

### Verification without lintrunner

```bash
cd agent_space/torch-xpu-ops-pr3383
# Reproduces the exact 6 CI errors against the unfixed file:
python3 tools/linter/adapters/grep_linter.py \
  --pattern=create_unbacked \
  --linter-name=META_NO_CREATE_UNBACKED \
  --error-name=test --error-description=test \
  -- test/xpu/dynamo/test_misc_xpu.py
```

After the `exclude_patterns` change, `lintrunner` (which reads `.lintrunner.toml`) skips the file entirely; the adapter is never invoked.

### Alternative strategies (rejected, for reference)

- **Narrow `include_patterns`** to `torch/_meta_registrations.py` like upstream — broader config change, affects intent of having the rule repo-wide. Ask the user.
- **Rewrite as `getattr(main, "create_unbacked_symint")()`** — avoids the literal string, but hacky and diverges from upstream.

### Key takeaway

For any `grep_linter.py`-backed rule, `# noqa` doesn't work. Fix via `.lintrunner.toml` scoping, not via comments.

## Example 4: Auto-Formatting vs Manual Fix

### Error Context
```bash
# test/xpu/dynamo/test_ctx_manager_xpu.py
# lintrunner wants formatting fix
```

### Analysis Approach

```bash
# Step 1: Run lintrunner to see automatic formatting
lintrunner -a test/xpu/dynamo/test_ctx_manager_xpu.py

# Step 2: Review diff - is it semantic or just formatting?
git diff test/xpu/dynamo/test_ctx_manager_xpu.py
# Shows reformatting of conditionals:
# OLD: if torch.cuda.is_available() and not torch.xpu.is_available()
# NEW: if (
#        not torch.cuda.is_available() and not torch.xpu.is_available()
#    ):
```

**Decision: ACCEPT AUTO-FORMATTING**
- These are PEP8/E501 compliance changes
- No semantic change to logic
- Improves code readability
- lintrunner intended for automatic application

### Correct commit style
```bash
# Commit auto-formatting separately from code fixes
git commit -m "Lint: apply auto-formatting fixes to dynamo xpu tests"
# vs
git commit -m "Lint: fix E731 triton grid lambda warnings"
```

## Example 5: Merging Multiple Lint Commits

### Scenario
After fixing E731 across multiple files AND applying formatting:
- Commit 1: fix E731 in test_aot_autograd_cache_xpu.py
- Commit 2: fix E731 in test_functions_xpu.py
- Commit 3: auto-formatting in 7 files

### Merge Process

```bash
# Step 1: Verify commit order
git log --oneline HEAD~3..HEAD
# bf528e22c0f Lint: apply auto-formatting fixes from lintrunner to dynamo xpu tests
# ed35f044317 Lint: fix E731 triton grid lambda warnings in dynamo test files
# 4ef83fc3199 Lint: skip E731 warning for triton grid lambda in test_functions_xpu.py

# Step 2: Soft reset to first lint commit - 1
git reset --soft 4ef83fc3199~1

# Step 3: Verify staged changes
git status
# Should show all modified files from 3 commits

# Step 4: Create single atomic commit
git commit -m "Lint: apply lint fixes to PR #3383 dynamo xpu test files

- Add noqa: E731 to all triton grid lambda declarations in test files (required
  for triton kernel grid scheduling, can't use regular functions)
- Apply auto-formatting via lintrunner to ensure consistent code style

Files modified: test_aot_autograd_cache_xpu.py, test_functions_xpu.py,
test_compiler_bisector_xpu.py, test_ctx_manager_xpu.py,
test_misc_xpu.py, test_regional_inductor_xpu.py,
test_streams_xpu.py, test_wrap_inductor_compiled_regions_xpu.py,
test_aotdispatch_xpu.py

9 files changed, 189 insertions(+), 77 deletions(-)"
```

### Push to correct remote

```bash
# For intel PR
git push intel HEAD:pr-3383

# For daisyden fork  
git push daisyden HEAD:daisyden/feature_branch --force
```

## Key Takeaways

1. **Deep analysis > pattern matching**: read the adapter code and the rule's include/exclude scope, not just the error line.
2. **Not all linters honor `# noqa`**: `grep_linter.py` without `--allowlist-pattern` does not. Adding noqa comments there is dead code.
3. **Fix at the right layer**: code edit, `# noqa`, or `.lintrunner.toml` scope — pick based on the adapter and the rule's intent.
4. **Triton `grid = lambda meta: ...`**: E731 is flake8; `# noqa: E731` works on the `lambda` line.
5. **Expected outputs need noqa on closing `"""`**: for B950 inside `assertExpectedInline` strings.
6. **Ported tests != meta registrations**: when an upstream test is ported into torch-xpu-ops, check whether the rule scope differs from pytorch's.
7. **Verify without lintrunner**: invoke the adapter directly to reproduce and confirm fixes.
8. **Commit only when explicitly asked**, and push to the PR head branch on the fork, never to `main`.