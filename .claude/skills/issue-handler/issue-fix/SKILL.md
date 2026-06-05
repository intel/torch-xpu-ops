---
name: issue-fix
description: >
  Fix a triaged CI failure. The issue body contains root cause analysis
  and proposed fix strategy — follow them. Works for both pytorch and
  torch-xpu-ops repos, UT and E2E failures.
---

# Fix CI Failure

## Step 0: Verify Environment
1. Check which repo you're in: `basename $(git rev-parse --show-toplevel)`
   - If `torch-xpu-ops`: you're fixing XPU kernel/operator code (files under `src/`)
   - If `pytorch`: you're fixing PyTorch core code (files under `torch/`, `aten/`, `test/`, `c10/`)
2. Verify the worktree is clean: `git status` should show no uncommitted changes
3. Start from a clean base: checkout the main branch (or your reproducer branch)

## Step 1: Reproduce
Extract the reproducer command from the issue description. It may be:
- A pytest command
- A python script
- A bash command
- Any other test invocation

Run it exactly as specified. If the issue has no reproducer, construct one
from the failed test name and error context.

If you modified C++/CUDA/SYCL code (not just Python), rebuild first:
- **pytorch repo**: `source ~/intel/oneapi/setvars.sh --force 2>/dev/null && git submodule sync && git submodule update --init --recursive && TORCH_XPU_ARCH_LIST=pvc USE_XPU=1 pip install -e . -v --no-build-isolation 2>&1 | tail -20`
- **torch-xpu-ops repo**: No separate build step needed (built as part of pytorch)

### torch-xpu-ops submodule pin (xpu.txt)
pytorch pins torch-xpu-ops at a specific commit via `third_party/xpu.txt`.
During CMake build, pytorch reads this SHA, fetches the torch-xpu-ops repo,
and checks out that exact commit into `third_party/torch-xpu-ops/`.

**To test a torch-xpu-ops fix inside pytorch:**
1. Copy only the changed files into `$PYTORCH_DIR/third_party/torch-xpu-ops/`
2. Run `ninja -C $PYTORCH_DIR/build` for an incremental rebuild (only recompiles changed files)
3. Run tests from the pytorch root directory
4. After testing, restore: `cd $PYTORCH_DIR/third_party/torch-xpu-ops && git checkout .`

**Do NOT** do a full `git checkout <branch>` in `third_party/torch-xpu-ops/` —
this changes mtimes on all files and triggers a massive ninja rebuild.
Copy only the changed files to keep incremental builds fast.
**Do NOT** modify `third_party/xpu.txt` — changing the pin triggers CMake
reconfiguration and a full rebuild from scratch (~hours).

## Step 2: Implement the Fix

### Coding Principles

1. **Think Before Coding** — State assumptions explicitly. If multiple interpretations exist, present them. If something is unclear, stop and ask.
2. **Simplicity First** — Minimum code that solves the problem. No speculative features, no abstractions for single-use code, no "flexibility" that wasn't requested.
3. **Surgical Changes** — Touch only what you must. Don't "improve" adjacent code, comments, or formatting. Match existing style. Remove imports/variables that YOUR changes made unused, but don't clean up pre-existing dead code.
4. **Goal-Driven Execution** — Define verifiable success criteria. Loop until verified.

### Key Rules
- **Minimal changes** — fix only what's broken
- **Never skip tests** — no `@skipIfXpu`, `@skip`, `unittest.skip`.
- **Stay in your repo** — if in pytorch, don't modify `third_party/*` (exception: you may edit `third_party/torch-xpu-ops/` files when the issue targets torch-xpu-ops sources — see HARD RULES).
- **Never modify unrelated files**

### Fix Strategies by Category
- **Unit tests (non-E2E):** For UT failures (not end-to-end models), see the **UT Skip Removal** section below.
- **Newly added test:** Try to enable it for XPU. If XPU support is genuinely missing and out of scope for this fix, do NOT add skip decorators — instead, add comments in the issue body and report `NEEDS_HUMAN` with reason "Requires new feature support, cannot fix in current scope".
- **Regression:** Find the guilty commit by reviewing recent commit history. Apply an XPU-specific fix if necessary. If you can't identify the guilty commit, compare with cuda/rocm backend to find the root cause.
- **Tolerance:** Match upstream CUDA tolerances when adjusting XPU tolerances.
- **Skip decorator stale:** See the **UT Skip Removal** section below.

### UT Skip Removal

When the fix is removing a stale `@skipIfXpu` / `@xfailIfXPU` / `@expectedFailureXPU` decorator:

**1. Find skip markers** — scan for these patterns:

| Pattern | Location |
|---------|----------|
| `@skipXPU`, `@unittest.skipIf(..., "XPU ...")` | Decorator on test method |
| `@expectedFailureXPU`, `@xfailIfXPU` | Decorator on test method |
| `DecorateInfo(unittest.skip("..."), device_type='xpu')` | Inside `OpInfo` definitions |
| Skip dictionaries: `skip_xpu`, `xfail_xpu` | Used in `instantiate_device_type_tests` |
| `skipIfXpu` in conditional blocks | Inline skip logic |

```bash
grep -n "skipXPU\|skipIfXpu\|xfailIfXPU\|expectedFailureXPU\|device_type='xpu'" test/<test_file>.py
grep -n -A2 "DecorateInfo.*skip.*xpu" torch/testing/_internal/common_methods_invocations.py
```

**2. Remove the marker** — delete the decorator/entry. Clean up unused imports if the decorator was the last usage. For `OpInfo` `DecorateInfo` entries, remove the entry from the `decorators` list.

**3. Verify locally** — run the test on XPU. If it **FAILS**, the underlying bug is not fixed — do NOT remove the skip. If it **PASSES**, proceed.

**4. Dynamic test names** — many test classes are dynamically generated via `instantiate_device_type_tests` (e.g., `TestCommonXPU` from `TestCommon`). If simple search fails, check for the base class + device suffix pattern.

## Step 3: Verify
Run the reproducer command again to confirm the fix works.
Also run related tests to check for regressions.

If you modified C++/CUDA/SYCL code, rebuild pytorch before verifying.

## Step 4: Clean Up
```bash
# Stage only your changes (exclude third_party, submodules)
git add <your_files>
git diff --cached --stat  # verify only intended files
```

## Step 5: Update Issue Description
After fixing, update the issue body with:
- What was changed and why
- Test results (pass/fail)

## HARD RULES
- NEVER add skip decorators. FIX the test.
- NEVER modify files outside your repo scope. Exception: if you're in the pytorch repo and the issue targets `third_party/torch-xpu-ops/` files, you may edit those (they are torch-xpu-ops source files bundled as a submodule). Do NOT modify other `third_party/*` submodules. If the fix is in `torch-xpu-ops`, you are supposed to add in the issue body of the instructions of the fix.
- NEVER modify unrelated files.
- NEVER force-push. This makes commits un-trackable.
- Use `git add` on specific files only.

## Best Practices & Pitfalls
- Always reproduce before fixing.
- **After fixing, run EVERY failing test case from the report individually.** Do not skip any case or assume verifying one representative case is sufficient.
- Match upstream CUDA tolerances when adjusting XPU tolerances.
- Remove unused imports when removing skip decorators.
- Keep commits focused: one fix per commit.
- **Never cherry-pick** upstream fixes. If a fix already landed on trunk, rebase (`git rebase origin/main`) instead.
- **Always rebuild after rebase or branch switch.** After `git rebase`, `git checkout`, or any operation that changes the commit base, rebuild (`source ~/intel/oneapi/setvars.sh --force 2>/dev/null && git submodule sync && git submodule update --init --recursive && TORCH_XPU_ARCH_LIST=pvc USE_XPU=1 pip install -e . -v --no-build-isolation`) before running tests. Without rebuilding, C++ extensions are stale and results are unreliable.
- Editable installs resolve Python from source but C++ headers from the installed location (`torch/include/`). After editing a C++ header, **manually copy** it to the installed include path.
- Delete the PCH cache (`/tmp/torchinductor_<user>/precompiled_headers/`) after modifying any header under `torch/csrc/inductor/cpp_wrapper/` — stale precompiled headers mask the fix.
- For C++ compile errors in AOT Inductor generated code (`CppCompileError`), the root cause is usually in the **codegen ordering** in `cpp_wrapper_cpu.py` (e.g., a function used before its definition is emitted). Check `write_wrapper_decl()` and `generate_input_output_runtime_checks()` ordering.

## Output
At the end, output:
```
### Agent Summary
- **What I found:** <root cause in one sentence>
- **What I changed:** <bullet list of files>
- **Test result:** <PASS/FAIL with test command>
- **Open questions / risks:** <concerns or "None">
```
