---
name: develop-xpu-test
description: Enable Intel XPU coverage for an already device-generic PyTorch test file. Use after a test has been refactored to be accelerator-agnostic and you now need to add XPU to its device instantiation and register XPU xfails/skips in op_db. Gates on review-test-refactoring first (hard stop if not clean), then extends instantiate_device_type_tests to include "xpu"/adds a HAS_GPU guard (always inline the tuple, never a separate variable), mirrors largeTensorTest and dtype decorators for XPU, and widens DecorateInfo device_type entries in common_methods_invocations.py ONLY for entries referencing the target class/its generic test names (untouched when no match). Never touches unrelated classes/names or inline device-conditionals, never edits test method body logic even to fix an obvious bug (reports it instead), and never adds skips (handled by enable-xpu-test orchestrator). Hands off to verify-xpu-test after edits.
---

# Develop XPU Test

Enable Intel XPU backend coverage on a PyTorch test that is *already*
device-generic (accelerator-agnostic). Exactly three edit types, nothing else:

1. **Instantiation enablement** — add `"xpu"` to the class's device
   instantiation (or a `HAS_GPU` guard for surfaces without
   `instantiate_device_type_tests`).
2. **Decorator parity** — mirror CUDA-only `largeTensorTest`/dtype decorators
   to XPU with identical scope.
3. **op_db widening** — extend `DecorateInfo(device_type=...)` entries in
   `common_methods_invocations.py` so CUDA-registered expected-failures/skips
   also apply on XPU.

Nothing else is in scope. In particular, this skill never adds a new skip
(decorator or inline), never touches an existing XPU skip, and never edits a
test method's body logic — see Constraints.

Local test/verification is out of scope — run `verify-xpu-test` after editing.

## When to Use

- A test class is already accelerator-agnostic (`device` parameter,
  `instantiate_device_type_tests`, `@onlyAccelerator` not `@onlyCUDA`) and you
  want XPU turned on for it.
- Following up the `test_ops.py`/`op_db` pattern: widen `device_type='cuda'`
  to `device_type=('cuda', 'xpu')`.

Do **not** use this to generalize a CUDA-only test (rename classes, replace
`torch.cuda.*`, swap `@onlyCUDA` → `@onlyAccelerator`). That must already be
done and reviewed — this skill assumes it and gates on it in Step 1.

## Tools Used

- **Read / grep / glob**: inspect the test file and `common_methods_invocations.py`.
- **edit**: instantiation and op_db changes only.
- **task (subagent)**: run the `review-test-refactoring` gate.

## Workflow

### Step 1: Review Gate (HARD STOP)

Dispatch `review-test-refactoring` against the target file (or diff/branch):

```python
task(
    subagent_type="explore",
    load_skills=["review-test-refactoring"],
    description="Review gate for XPU enablement of <test_file>",
    prompt="Review <ABSOLUTE_PATH> against the decoupling standards. Report "
           "Blockers/Major/Minor. Return whether it is clean enough to enable "
           "XPU on (zero Blockers).",
)
```

**PASS** = zero Blockers (Majors/Minors surfaced but don't block). **FAIL** =
any Blocker. On FAIL: make zero edits, report the Blocker/Major findings
verbatim, state enablement is halted until fixed, and stop — do not proceed
to Step 2.

### Step 2: Enable the Test Instantiation

#### 2.1 Pattern A — `instantiate_device_type_tests` (preferred)

```python
instantiate_device_type_tests(
    TestSDPAGpuOnly, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)
```

- `only_for=("cuda",)` → `only_for=("cuda", "xpu")` + `allow_xpu=True`.
- `only_for=("cpu",)` → no change needed.
- No `only_for=` but missing `allow_xpu=True` → add `allow_xpu=True`.
- Keep any existing `except_for=`/`allow_mps=` intact.
- **Always inline the tuple directly** (`only_for=("cpu", "cuda", "xpu")`). Never introduce a separate named variable (e.g. `only_for_xpu`). This keeps the diff minimal and avoids clutter in the test file.

#### 2.2 Pattern B — `HAS_GPU` guard

For raw-availability gates instead of parametrized device types:

```python
HAS_GPU = torch.cuda.is_available() or torch.xpu.is_available()

@unittest.skipIf(not HAS_GPU, "CUDA or XPU is unavailable")
```

Replace the CUDA-only guard (e.g. `@unittest.skipIf(not torch.cuda.is_available(), ...)`)
with the `HAS_GPU` form. Use exactly one of Pattern A/B, matching file style.

### Step 3: Decorator Parity

**`largeTensorTest`**: `@largeTensorTest("20GB", "cuda")` → add
`@largeTensorTest("20GB", "xpu")` with the identical size string.

**dtype decorators**: `@dtypeIfCuda(torch.float32, torch.float64)` → add
`@dtypesIfXpu(torch.float32, torch.float64)` with identical scope. Import
`dtypeIfXpu`/`dtypesIfXPU` from `torch.testing._internal.common_device_type`
if not already present. Match existing file style for ordering/placement;
add only the missing XPU decorator/import.

### Step 4: Update `op_db` DecorateInfo Entries

> **SCOPING RULE:** Only widen entries that reference the **target class**
> and its **actual generic test names**. Never widen entries for other
> classes/names (`TestForeach`, `TestCommon`, ...) — that's a bug, not a fix.
> **Zero matching entries ⇒ zero edits to `common_methods_invocations.py`.**

#### Step 4.0: Determine in-scope (test_class, test_name) pairs — MANDATORY

1. **Target class name** = the base class name enabled in Step 2 (not the
   device-suffixed instantiated name, e.g. match `TestComplexTensor` not
   `TestComplexTensorXPU`).
2. **Generic test names** = every `test_*` method the class defines (read the
   class body).
3. **Grep for matches**:
   ```
   grep -nE "['\"]<TestClassBaseName>['\"]" torch/testing/_internal/common_methods_invocations.py
   grep -nE "['\"]<test_name>['\"]"        torch/testing/_internal/common_methods_invocations.py
   ```
   In scope only if BOTH the `test_class` and `test_name` strings match (a
   class-only match, i.e. no test name specified, applies to the whole class
   and is sufficient).
4. **Zero matches** → skip the rest of Step 4; state explicitly in the report
   that no op_db change is needed. **One or more matches** → apply the
   transforms below to those entries only; leave everything else untouched.

Do not widen an entry just because it's a `device_type='cuda'` complex-op
entry — class/name match is required, no exceptions.

**Transform 1 — widen `'cuda'` to the tuple form** (applies to
`unittest.skip(...)`, `unittest.expectedFailure`, `toleranceOverride(...)`;
preserves `active_if=`/`dtypes=`/class/name strings):

```python
# BEFORE
DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cuda'),
# AFTER
DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type=('cuda', 'xpu')),
```

**Transform 2 — collapse duplicate cuda + xpu lines** for the same
`(test_class, test_name, dtypes, active_if)` into one tuple entry:

```python
# BEFORE (two lines)
DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cuda'),
DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='xpu'),
# AFTER
DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type=('cuda', 'xpu')),
```

**Never touch**: `device_type='mps'`/`'cpu'` entries, or entries already
`('cuda', 'xpu')`. Wrap onto a continuation line if `('cuda', 'xpu')` pushes
past linter width, matching surrounding style.

## Constraints

1. **Review gate is a hard stop.** Any Blocker ⇒ zero edits, report, stop.
2. **This skill never adds skips.** Adding `@skipIfXpu` is handled by the
   `enable-xpu-test` orchestrator in its Phase 5B, after `check-known-issue`.
   This skill only performs three edit types: instantiation, decorator parity,
   and op_db widening. Prohibited under all circumstances: `@skipXPU`,
   `@skipXPUIf`, `@skipIfXpu`, `subtest(..., decorators=[skipIfXpu(...)])`,
   `self.skipTest("xpu", ...)`, and inline device-conditionals written as skip
   substitutes (e.g. `if dtype is torch.float and device in ("cpu", "cuda"):`).
3. **Never touch existing XPU skips/decorators** (`@skipIfXpu`, `skipXPU`,
   `@skipXPUIf`, pre-existing `device_type='xpu'` entries not part of a
   Transform 2 merge) — leave exactly as-is.
4. **Never edit test method body logic**, even for an obviously-correct
   one-token fix. Steps 2-4 are the only edits in scope. If enabling XPU
   surfaces (or is expected to surface) a body-level bug — e.g. a
   `torch.randint(...)` call missing `device=`, previously masked by a
   device-specific early-return that never fired for XPU — do not patch it
   and do not add a skip to route around it (this is a test-code bug, not a
   backend gap). Proceed with Step 2 normally; report the bug (file, line,
   current code, suggested one-line fix labeled "not applied by this skill")
   alongside the summary, and let `verify-xpu-test`'s resulting failure route
   it to a dedicated fix (`fix-ut-test-code` skill, or explicit user request)
   — never bundle it into this skill's edits.
5. **Only widen in-scope op_db entries** (Step 4.0); no cherry-picking within
   scope, no touching entries for other classes/names even if `cuda`-typed.
   Zero matches ⇒ zero edits to `common_methods_invocations.py` — this is the
   correct, common outcome, not a failure to fix. Before handing off, re-read
   the diff and confirm every changed `DecorateInfo` traces to the target
   class/its test names; revert any that don't.
6. **Never touch `mps`/`cpu` DecorateInfo entries.**
7. **No local verification here.** Do not run pytest — hand off to
   `verify-xpu-test`.
8. **No commits unless asked.** Present the summary + `verify-xpu-test`
   result; commit only on explicit request.
9. **ASCII only**; match existing file style.
10. **Decorator parity is mechanical** — mirror `largeTensorTest`/dtype
    decorators with identical scope; never alter dtype sets or size
    thresholds while adding parity.
11. **Never cite a CLOSED issue as gating justification.** Before treating any
    GitHub issue as evidence for a skip/xfail/gate, check
    `gh issue view <n> --repo <owner>/<repo> --json state,stateReason`. Only
    `OPEN` counts. A `CLOSED` issue — even titled `[Bug Skip]: ...` — usually
    means the gap was already resolved through the project's own mechanism
    (e.g. an entry in `torch-xpu-ops/test/xpu/skip_list_common.py`); citing it
    to justify new source-level gating both misrepresents the issue as active
    and duplicates an existing resolution. If a closed issue is the only
    evidence for a gap, re-run the test to verify current behavior instead of
    trusting the issue's title.

## Worked Example (Constraint 4)

Enabling `TestFoo` surfaces (or is expected to surface):
`RuntimeError: Expected all tensors to be on the same device, but found at
least two devices, xpu:0 and cpu!`, caused by `test_bar`'s
`torch.randint(high=size, size=(n,))` lacking `device=`, previously masked by
`if self.device_type == "cuda": return` above it.

Do: apply Step 2's instantiation edit normally for the whole class, then
report — "`test_bar` (test/test_foo.py:123) has a latent device-mismatch bug:
`torch.randint(...)` always creates a CPU tensor; suggested fix (NOT applied
here): add `device=device`. Route to a dedicated fix separately."

Do not: add `device=device` yourself, or gate `test_bar` with a skip.

## Summary of Edits Produced

- `test/<file>.py`: instantiation extended to XPU (Pattern A/B) + decorator
  parity mirrored. Body logic untouched — a body-level bug is reported, never
  patched or gated around (Constraint 4).
- `common_methods_invocations.py`: **only when Step 4.0 finds in-scope
  entries** — widened/collapsed to `('cuda', 'xpu')`. No new XPU skips, no
  changes to existing ones, no changes to other classes/names. Zero matches
  ⇒ file left untouched.

## See Also

- `enable-xpu-test` — orchestrator that consumes these edits, verifies them,
  and handles failure follow-up (including `@skipIfXpu` addition for known
  backend gaps via `check-known-issue`).
- `verify-xpu-test` — local XPU verification of these edits.
- `submit-xpu-test-pr` — packages verified edits into a confirm-gated draft PR.
- `review-test-refactoring` — the Step 1 gate.
