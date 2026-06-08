---
name: xpu-nightly-ci-fix
description: Analyze CI nightly test failures and fix XPU test cases. Use when the user provides CI failure reports, nightly status emails, or lists of failing test cases. Covers triaging failures, reproducing locally, identifying root causes, and applying fixes.
---
> Please also read and refer to `CLAUDE.md` at the repository root for general behavioral guidelines before proceeding.

# Nightly CI UT Fixing for XPU

## When to Use

- The user provides a CI nightly failure report (email content, test case list, or log snippets).
- The goal is to analyze, triage, reproduce, and fix multiple failing XPU test cases in PyTorch, then output a summary report.

## Step 1: Parse the failure report
- Parse the Pytorch commit_id, report_date.
- Extract the list of failing test cases (test file, class, method name)
- Group failures by test file / module

### Step 2: Reproduce locally

- fetch latest pytorch origin main if commit_id is provided in previous step, checkout to it else use origin main.
- Create a new branch for the fix, branch name: fix-<report_date>
- build pytorch:
  ```bash
  source build_pytorch.env
  python setup.py clean
  pip install -e . -v --no-build-isolation
  ```

build_pytorch.env.example, adjust the content as needed for your local environment and save as `build_pytorch.env`:
```
export TORCH_XPU_ARCH_LIST=pvc
export USE_XPU=1
export USE_CUDA=0

# Adjust these paths to your local oneAPI installation
source /opt/intel/oneapi/compiler/latest/env/vars.sh
# Check if oneapi-vars.sh is present, sometimes it's located directly in oneapi/
# source /opt/intel/oneapi/setvars.sh

# pti is optional depending on the scenario
# source /opt/intel/oneapi/pti/latest/env/vars.sh
```

- Run each failing test on this machine:
  ```bash
  source build_pytorch.env && python <test_file> -k <test_name> 2>&1 | tail -80
  ```
- Confirm the failure reproduces.


### Step 3: Analyze and categorize

For each failure, determine the root cause category:
- First, use git to check when the test case was added. If it was added recently, identify which commit or PR introduced it. If it's newly added, check the commit or PR for relevant changes, to see if xpu support is required. then go to fix step. otherwise proceed to categorize the failure.
1. **XPU backend bug** — Fix in `torch/_inductor/` or `third_party/torch-xpu-ops/`
2. **Tolerance too tight** — Increase atol/rtol to match CUDA tolerances
3. **Skip decorator stale** — Remove `@skipIfXpu` or `@expectedFailure` if the test now passes
4. **Community Upstream regression** — New upstream code broke XPU; needs XPU-specific workaround
5. **Test infrastructure** — Environment, import, or setup issue

### Step 4: Fix

- For newly added test case: try to enable it for XPU. If it cannot be enabled directly, leave it skipped with a proper reason, add a TODO with issue ID, and file a tracking issue in `intel/torch-xpu-ops` for engineering follow-up.
- For a regression test, first try to find the guilty commit by reviewing the recent commit history and identifying which change introduced the failure. Apply an XPU-specific fix if necessary.
- If can not identify the guilty commit, start analysis, compare with cuda/rocm backend, and try to find the root cause.


### Step 5: Verify

- Run the fixed test and confirm it passes
- Run the full test file to check for regressions
- Linter check:
```bash
spin fixlint
```
- Analyze the output log and fix any linting issues.
- Stage and commit changes. The commit message needs to include: title `[xpu][fix] <description>`, `## Motivation`, `## Solution`, `## Test plan`.

### Step 6: Generate Summary Report

For all the failing test cases provided in the initial CI report, write a structured summary to `summary_<date>.md`.

Each entry in the summary should include:
- Test case name / module
- Root cause and category (from Step 3)
- Resolution or fix applied (with commit reference if a change was made)
- AR: Action required (e.g., submit PR, rebase, none)

## Environment

- Source `build_pytorch.env` before running any test
- Build: `pip install -e . -v --no-build-isolation`
- Scratch files go in `agent_space/` (git-ignored)

## Best Practices

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.


**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.


**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.


**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.


**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
- Always reproduce before fixing
- **After fixing, run EVERY failing test case from the report individually to verify it passes.** Do not skip any case or assume that verifying one representative case is sufficient. Run all cases explicitly, batch by batch if needed, and confirm each one passes.
- Match upstream CUDA tolerances when adjusting XPU tolerances
- Remove unused imports when removing skip decorators
- Keep commits focused: one fix per commit
- **Never cherry-pick** upstream fixes into the fix branch. If a fix already landed on trunk after the CI commit, rebase the fix branch onto the latest trunk (`git rebase origin/main`) instead.
- **Always rebuild after rebase or branch switch.** After `git rebase`, `git checkout`, or any operation that changes the commit base, you MUST rebuild (`source build_pytorch.env && pip install -e . -v --no-build-isolation`) before running any tests. Without rebuilding, the installed C++ extensions and generated code are stale and test results are **completely unreliable** — they may segfault, produce wrong pass/fail results, or mask real issues. Never trust test results from a stale build.
- Editable installs resolve Python from source but C++ headers from the installed location (`torch/include/`). After editing a C++ header, **manually copy** it to the installed include path.
- Delete the PCH cache (`/tmp/torchinductor_<user_name>/precompiled_headers/`) after modifying any header under `torch/csrc/inductor/cpp_wrapper/` — stale precompiled headers mask the fix.
- For C++ compile errors in AOT Inductor generated code (`CppCompileError`), read the generated `.wrapper.cpp` error message carefully — the root cause is usually in the **codegen ordering** in `cpp_wrapper_cpu.py` (e.g. a function used before its definition is emitted). Check `write_wrapper_decl()` and `generate_input_output_runtime_checks()` ordering.


## Requirements

- PyTorch built from source with XPU support
- `build_pytorch.env` file configured for the oneAPI environment
