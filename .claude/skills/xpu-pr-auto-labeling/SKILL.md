---
name: xpu-pr-auto-labeling
description: >
  Rules for automatically determining which disable_* labels to apply to a PR
  based on the file paths changed. Used by the auto-label workflow.
---

# PR Auto-Labeling Rules

## Purpose

Determine which `disable_*` CI labels to apply to a PR based on the files changed.
These labels control which CI jobs run, reducing unnecessary CI load.

## Available Labels

| Label | Effect |
|-------|--------|
| `disable_all` | Skip all CI test jobs, only run lint |
| `disable_ut` | Skip non-distributed UT jobs |
| `disable_e2e` | Skip e2e (inductor benchmark) jobs |
| `disable_distributed` | Skip distributed UT jobs |
| `disable_win` | Skip Windows CI jobs |
| `disable_build` | Skip source build, use nightly wheel |

Note: Only labels consumed by `pull.yml` are in scope. Other labels (`disable_accelerate`, `disable_transformers`) are used by separate workflows and not managed here.

## Decision Rules

Evaluate the file paths changed in the PR and apply labels using the following logic.
Rules are evaluated top-to-bottom; use the FIRST matching rule set.

### Rule 1: Pure CI infrastructure (no functional logic change)

**Condition:** ALL changed files match `.github/workflows/`, `.github/scripts/`, `.github/ISSUE_TEMPLATE/`, `.github/copilot-instructions.md`, `.claude/skills/`, or other `.github/` non-workflow metadata files AND the workflow changes do NOT alter job execution logic (only change triggers, permissions, concurrency, comments, labels conditions, yaml formatting).

**Labels:** `disable_all`

**Examples:** Fixing lint workflow permissions, updating issue templates, adding skills docs.

### Rule 2: CI workflow with execution logic changes

**Condition:** Changed files include `.github/workflows/` AND the changes alter actual job steps, scripts, environment, or build/test commands.

**Labels:** Disable all jobs EXCEPT the ones whose workflows are being modified. The modified workflow's jobs must run to validate the change.

- If modifying `_linux_build.yml` → keep linux-build running
- If modifying `_linux_ut.yml` → keep linux-ut running
- If modifying `_linux_e2e.yml` → keep linux-e2e running
- If modifying `_windows_*.yml` → keep windows jobs running, add `windows_ci` to force trigger

**Additional:** If no `src/`, `cmake/`, `CMakeLists.txt` files changed, add `disable_build`.

### Rule 3: Only skip/expect list changes

**Condition:** ALL changed files are `test/xpu/skip_list_common.py`, `test/xpu/expect/`, or `test/xpu/test_decomp_xpu.py` (pure skip list / expect file updates with no test logic change).

**Labels:** `disable_all`

**Rationale:** Skip list changes don't need functional validation beyond lint.

### Rule 4: Test-only changes

**Condition:** ALL changed files are under `test/` (excluding pure skip/expect files handled by Rule 3).

**Labels (base):** `disable_e2e`, `disable_distributed`

**Additional modifiers:**
- If files are ONLY under `test/xpu/functorch/` or `test/xpu/dynamo/` or `test/xpu/higher_order_ops/`: add `disable_win`
- If NO files under paths covered by Windows UT (`test/xpu/test_ops*.py`, `test/xpu/core/`, `test/xpu/test_nn*.py`): add `disable_win`

**Additional:** Always add `disable_build` (test-only changes don't need source build).

### Rule 5: Kernel/operator source changes (src/)

**Condition:** Changed files include `src/ATen/` or `src/xccl/`.

**Labels:** Depends on which subdirectory:

- `src/ATen/` only (kernel/operator code, no xccl):
  - Base: `disable_e2e`, `disable_distributed`
  - Do NOT add `disable_win` — kernel fixes may be platform-specific (e.g. a fix specifically for Windows)

- `src/xccl/` involved (distributed communication backend):
  - Base: `disable_e2e` only — do NOT add `disable_distributed` (xccl changes must run distributed tests)
  - Add `disable_win` (xccl is Linux-only)

**Note:** Do NOT add `disable_build` — source changes require compilation.

### Rule 6: Build system changes

**Condition:** Changed files include `cmake/`, `CMakeLists.txt`, `src/**/*.cmake`, or `tools/`.

**Labels:** None (or minimal). Build system changes need full CI validation.

**If ONLY build system files changed** (no test, no kernel): `disable_e2e`, `disable_distributed`

### Rule 7: yaml/ definition changes

**Condition:** Changed files include `yaml/` (operator definitions).

**Labels:** Minimal — operator definition changes can affect many things. Run full CI.

Apply: `none` (run full CI by default).

### Rule 8: Mixed changes (fallback)

**Condition:** None of the above rules fully apply (mixed src + test + CI, etc.).

**Labels:** Apply the INTERSECTION of what each individual file category would disable.

- If any `src/` file changed: do NOT add `disable_build`
- If any `cmake/`/`CMakeLists.txt` changed: do NOT add `disable_build`
- Only disable a job if ALL changed files agree it should be disabled

## disable_build Logic (cross-cutting)

`disable_build` can be added whenever ALL of these are true:
- No files changed in `src/` (excluding `src/**/*.py`)
- No files changed in `cmake/`
- No `CMakeLists.txt` changed
- No files changed in `tools/` that affect the build

When `disable_build` is applied, CI uses nightly wheel instead of building from source.

## Output Format

When used by the auto-label workflow, output ONLY the label names as a comma-separated list:

```
disable_e2e,disable_distributed,disable_win
```

If no labels should be applied (full CI needed), output:

```
none
```

## Important Notes

1. **`disable_auto` label** — When manually added to a PR, auto-labeling is completely disabled for that PR. The workflow will skip without allocating a runner. This label should only be added manually by humans (never by automation) to opt out of auto-labeling when full manual control over CI labels is needed.
2. **PRs with `ai_generated` label** should still follow these rules (they need CI validation).
3. **When in doubt, disable less.** It's better to run unnecessary CI than to miss a regression.
