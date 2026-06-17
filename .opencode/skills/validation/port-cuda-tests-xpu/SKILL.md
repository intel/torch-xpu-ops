---
name: port-cuda-tests-xpu
description: Port PyTorch CUDA test files to support Intel XPU backend testing. Use when generalizing CUDA-specific tests to run on XPU, enabling GPU tests for both CUDA and XPU backends, or when user mentions porting CUDA tests to XPU or enabling XPU test coverage.
---

# Port CUDA Tests to XPU Backend

This Skill provides a workflow for generalizing PyTorch CUDA-specific test files to support Intel XPU backend testing.

## Skill Integration

**This skill follows agent-guidelines AND extends it with specific constraints.**

Always apply agent-guidelines rules including:
- Mandatory post-write commit protocol (ask user before committing)
- Deep semantic analysis instead of pattern matching
- Atomic commits for each ported test
- All constraints defined in agent-guidelines

## Tools Used in Workflow

This skill relies on the following tools:
- **Read**: View file contents and understand current implementation
- **grep**: Search for patterns (cuda/xpu-specific code, test names, etc.)
- **glob**: Find test files by pattern matching
- **edit**: Make targeted changes to Python files
- **bash**: Run tests, git operations, check environment, check API validity
- **write**: Create or update files including SKILL.md and source code
- **curl**: Create GitHub PRs via REST API
- **task (explore)**: Explore codebase for similar patterns (quick/medium/thorough modes)
- **question**: Ask user for confirmation before PR submission
- **webfetch**: Fetch PyTorch documentation for API verification

## Preconditions

Before starting, verify:
1. The test file does NOT already exist in `torch-xpu-ops/test/xpu/` directory
2. No equivalent XPU-specific test exists in the torch-xpu-ops package

```bash
# Check if test already exists in torch-xpu-ops
ls torch-xpu-ops/test/xpu/ 2>/dev/null | grep -i <test_name> || echo "Not found - OK to port"
```

## Workflow

### Step 1: Analyze Test File for CUDA-Specific Code

Search for CUDA-specific patterns:
- `torch.cuda.is_available()`, `torch.cuda.device_count()`
- `torch.cuda.Event`, `torch.cuda.ipc_*`, `torch.cuda.stream`
- `device="cuda"` strings
- `TEST_CUDA_IPC` conditionals
- `@unittest.skipIf(not torch.cuda.is_available(), ...)`

```bash
grep -n "cuda" test_file.py | grep -v "#"
```

Check torch-xpu-ops doesn't have it:
```bash
ls torch-xpu-ops/test/xpu/ 2>/dev/null | grep -i <test_name>
```

### Step 2: Identify Portable vs Non-Portable Tests

**Tests NOT portable to XPU:**
- CUDA IPC (CUDA-specific Inter-Process Communication) tests
- Tests using `torch.cuda.ipc_collect()`, `_share_cuda_()`, IPC handles
- CUDA memory caching allocator specific tests
- `_cudaMalloc` related functionality

**Tests WITH XPU Counterpart (POTENTIALLY portable):**
- `torch.cuda.Event` - has XPU counterpart `torch.xpu.Event`
- `torch.cuda.Stream` - has XPU counterpart `torch.xpu.Stream`

**Tests portable to XPU:**
- GPU-accelerated tests already checking `torch.xpu.is_available()`
- Tests using `instantiate_device_type_tests(allow_xpu=True)`
- Functions using TorchScript fuser "fuser1" that supports XPU
- Tests with runtime device selection (cuda if available else xpu)

**CPU tests: Skip porting** - CPU-based tests are already device-agnostic and don't need XPU-specific handling.

### Step 3: Apply XPU Generalization Pattern

**Commit incrementally.** The work is naturally two phases. **Commit between them.** If you do, you never need to restructure history.

- **Phase 1 = Step 3a (generalization only, no XPU tokens) → commit.**
- **Phase 2 = Step 3b (XPU enablement + skips) → commit.**

Result: two clean atomic commits, no `git reset --hard`, no backup branch, no tree-identical diff to babysit.

#### Step 3a: Phase 1 — Generalization (no XPU tokens)

Goal: remove CUDA-only assumptions without adding XPU. After this step the file MUST contain **zero `xpu` tokens** and `instantiate_device_type_tests` MUST still be `only_for=("cuda",)`. This proves Phase 1 is a no-op on CUDA.

**Allowed in Phase 1:**

1. **Class rename** to an accelerator-neutral name (e.g. `TestSDPACudaOnly` → `TestSDPAGpuOnly`); update the `instantiate_device_type_tests` call to use the new name but keep `only_for=("cuda",)` unchanged.

2. **`torch.cuda.X` → `device_mod.X`** via:
   ```python
   device_mod = getattr(torch, torch.device(device).type)
   props = device_mod.get_device_properties(device)
   device_mod.empty_cache()
   ```
   No-op on CUDA (`device_mod is torch.cuda`); ready for XPU later.

3. **`torch.accelerator` API** swaps where the test uses `torch.cuda.is_available()` / `torch.cuda.current_stream()` / etc. purely for plumbing (see the API reference below).

4. **Decorators**: Replace `@onlyCUDA` with `@onlyAccelerator` to allow tests to run on any accelerator (CUDA, XPU, etc.) without hardcoding specific backends.

5. **Method Signatures**: Add `device` parameter to test methods that previously used hardcoded `device="cuda"` internally, and pass it to tensor creation functions.

6. **Imports** that are accelerator-agnostic (e.g. `subtest`, `onlyAccelerator` from `torch.testing._internal.common_device_type`) if needed. Importing is fine; **using** `skipIfXpu` inside it is Phase 2.

**Forbidden in Phase 1:**

- ANY `xpu` token (`skipIfXpu`, `is_xpu`, `torch.xpu.*`, `XPUGraph`, `"xpu"` string literal, `HAS_XPU`).
- Extending `only_for=` to include `"xpu"`.
- `allow_xpu=True`.
- XPUGraph / accelerator-graph plumbing — that lands in Phase 2.

**End of Phase 1 — verify on CUDA, then commit.**

**Phase 1 local-test gate (collection + no-crash on the XPU host).** Phase 1 is a pure refactor: CUDA behavior MUST be unchanged. The local validation env `pytorch_opencode_env` is an **XPU host** — `torch.cuda.is_available()` is False there, so CUDA-only cells will **skip**, not run. That is expected and fine: the local gate is checking collection, imports, parametrize matrix, and absence of crashes — NOT CUDA numerical correctness (that is validated in CI on a CUDA host).

```bash
source ~/miniforge3/bin/activate pytorch_opencode_env
cd /tmp

# Sanity-check the env: XPU available, CUDA not.
python -c "import torch; print('cuda:', torch.cuda.is_available(), \
                              ' xpu:', torch.xpu.is_available())"
# Expect: cuda: False  xpu: True  (this is correct for pytorch_opencode_env)

# Run the renamed class. CUDA-only cells will skip; nothing should fail.
python -m pytest <repo>/test/<file>.py -v -k "<NewClassName>" --tb=short \
  | tee /tmp/phase1.txt
```

Phase 1 gate passes when **all** of the following hold:

- The renamed class is collected (parametrize matrix matches the pre-rename baseline if you have one).
- **Zero failures.** Only `passed` and `skipped` outcomes are acceptable — CUDA-only cells legitimately skip on the XPU host because `torch.cuda.is_available()` is False.
- No `ERROR` lines (collection / import / teardown errors).

Sanity check before committing:

```bash
git diff --cached test/<file>.py | grep -i xpu
# MUST return zero matches.
```

Commit message:

```
Generalize <ClassName> and remove <accelerator>-only assumptions

Rename TestSDPACudaOnly -> TestSDPAGpuOnly and replace direct torch.cuda.*
calls with `device_mod = getattr(torch, torch.device(device).type)`. Pure
refactor: class is still instantiated only_for=("cuda",) and CUDA behavior is
unchanged. Prepares the class for XPU enablement in a follow-up commit.

Authored by Claude.
```

#### torch.accelerator API (Verified)

```python
from torch import accelerator

# Check if accelerator is available
accelerator.is_available()  # -> bool

# Get current accelerator name
accelerator.current_accelerator()  # -> str (e.g., "xpu", "cuda")

# Get/set current device index
accelerator.current_device_index()  # -> int
accelerator.set_device_index(device_idx)  # or set_device_idx
accelerator.set_device_idx(device_idx)    # alias

# Get device count
accelerator.device_count()  # -> int

# Get device capability
accelerator.get_device_capability(device=None)  # -> dict with "key" and "value"

# Memory operations
accelerator.memory_allocated(device_index=None)  # -> int (bytes)
accelerator.max_memory_allocated(device_index=None)  # -> int

# Synchronization
accelerator.synchronize(device=None)  # -> None

# Stream operations
accelerator.current_stream(device=None)  # -> torch.Stream
accelerator.set_stream(stream)  # -> None
```

#### Phase 1 generalization patterns

```python
# BEFORE - CUDA specific
if torch.cuda.is_available():
    device = "cuda"
    # ... CUDA specific code

# AFTER (Phase 1) - accelerator-agnostic. Note: produces "cuda" on a CUDA
# host because that is still the only accelerator the class is instantiated
# for. NO `xpu` literal here.
if accelerator.is_available():
    device = accelerator.current_accelerator()
else:
    device = "cpu"
```

```python
# BEFORE - Hardcoded device string check (limiting to CUDA)
if device == "cuda":

# AFTER (Phase 1) - Generalize to any accelerator (or robust device type check)
if torch.device(device).type != "cpu":  # If logic applies to all GPUs/accelerators
# OR if backend-specific logic is strictly unavoidable:
if torch.device(device).type == "cuda":
```

```python
# BEFORE - Hardcoded device
x = torch.randn(3, 3, device="cuda")

# AFTER (Phase 1) - device derived from the parametrize axis or accelerator API
x = torch.randn(3, 3, device=device)  # `device` comes from device_type test parametrize
```

#### Step 3b: Phase 2 — XPU enablement + skips

Now (and only now) add XPU. Phase 2 contains **all and only** the XPU-touching changes.

1. **Extend the instantiation**:
   ```python
   instantiate_device_type_tests(
       TestSDPAGpuOnly, globals(), only_for=("cuda", "xpu"), allow_xpu=True
   )
   ```

2. **Accelerator-graph plumbing** (when the class uses CUDAGraph):
   ```python
   is_xpu = torch.device(device).type == "xpu"
   g = torch.xpu.XPUGraph() if is_xpu else torch.cuda.CUDAGraph()
   graph_ctx = torch.xpu.graph(g) if is_xpu else torch.cuda.graph(g)
   stream = torch.Stream(device=device) if is_xpu else torch.cuda.Stream()
   current = torch.accelerator.current_stream() if is_xpu else torch.cuda.current_stream()
   # broaden device-residency assertions:
   self.assertTrue(o.is_cuda or o.is_xpu)
   ```
   If a particular CUDAGraph-bound test cannot be generalized cleanly, leave it CUDA-only and skip it on XPU via `@skipIfXpu` — do NOT force-fit XPUGraph if torch-xpu-ops doesn't yet support that surface.

3. **Alternative HAS_GPU pattern** (when `torch.accelerator` isn't suitable for the surface):
   ```python
   HAS_GPU = torch.cuda.is_available() or torch.xpu.is_available()

   @unittest.skipIf(not HAS_GPU, "CUDA or XPU is unavailable")
   ```

4. **Mirror `skip_list_common.py`**: for every `(class, test_name[, param_id])` cell that `third_party/torch-xpu-ops/test/xpu/skip_list_common.py` skips for this class on XPU, add the matching in-place skip:
   - Whole method unsupported on XPU → `@skipIfXpu(msg="<reason>; tracked at intel/torch-xpu-ops#NNNN")` on the method.
   - Single parametrize cell unsupported → wrap the offending value in `subtest(value, decorators=[skipIfXpu(msg="...")])`.
   - Mid-body carve-out → `if self.device_type == "xpu": self.skipTest("<reason>; intel/torch-xpu-ops#NNNN")`.

   **Every XPU skip MUST cite a torch-xpu-ops tracking issue.** No bare `@skipIfXpu`.

5. **Skip-list parity invariant**: after Phase 2, the set of `(class, test_name, param_id)` cells skipped on XPU upstream MUST equal the set skipped by `skip_list_common.py` for that class. If you add a new skip upstream, also remove it from `skip_list_common.py` (and vice versa); CI must not double-skip silently.

**End of Phase 2 — verify on CUDA AND XPU, then commit.**

**Phase 2 local-test gate (XPU runs, CUDA cells skip cleanly).** Phase 2 must (a) make the new XPU instantiations either pass or skip with a citing reason, and (b) leave CUDA-only cells skipping cleanly on the XPU host (same outcome as Phase 1). Both axes run in the same `pytorch_opencode_env` — no remote host needed.

```bash
source ~/miniforge3/bin/activate pytorch_opencode_env
cd /tmp

# Full class run. CUDA-only cells will skip (host has no CUDA); XPU cells
# should either pass or skip with a tracking-issue reason.
python -m pytest <repo>/test/<file>.py -v -k "<NewClassName>" --tb=short \
  | tee /tmp/phase2.txt

# Targeted XPU-only filter, for fast iteration on XPU failures:
python -m pytest <repo>/test/<file>.py -v -k "<NewClassName> and xpu" --tb=short \
  | tee /tmp/phase2.xpu.txt
```

Phase 2 gate passes when **all** of the following hold:

- **No new failures vs Phase 1.** Diff `/tmp/phase2.txt` against `/tmp/phase1.txt`: the set of CUDA-cell skip reasons MUST be unchanged. Phase 2 only **adds** XPU rows; it must not flip any CUDA row from `passed`/`skipped` to anything else.
- **No unexplained XPU failures.** Every XPU failure must be either (i) fixed in this commit, or (ii) covered by a `@skipIfXpu` / `subtest(decorators=[skipIfXpu(...)])` / in-body `self.skipTest("xpu", ...)` that cites an `intel/torch-xpu-ops#NNNN` tracking issue. A failing XPU test with no skip and no fix is a release blocker — DO NOT commit.
- **At least one XPU row is exercised.** Confirm via `grep -E 'xpu.*PASSED|xpu.*SKIPPED' /tmp/phase2.xpu.txt`. An empty XPU axis means `only_for=("cuda", "xpu")` / `allow_xpu=True` is missing or not taking effect.
- **Skip-list parity check.** Compare the set of XPU-skipped cells against `third_party/torch-xpu-ops/test/xpu/skip_list_common.py` for this class; the two sets MUST match.

CUDA numerical behavior is validated by CI on a CUDA host, not by this local gate. The local gate's job is to prove the XPU enablement works AND that CUDA-only cells still skip cleanly (i.e. nothing about the Phase 2 edits broke their skip predicates or crashed during teardown).

If `torch.xpu.is_available()` returns `False` in `pytorch_opencode_env`, the env is broken (wrong wheel, missing driver, etc.) — fix the env, do NOT skip Phase 2 verification.

Commit message:

```
[XPU] Enable <ClassName> on XPU and skip cases unsupported by torch-xpu-ops

Extend instantiate_device_type_tests to only_for=("cuda", "xpu") with
allow_xpu=True. Add XPUGraph plumbing for graph-bound tests. Add
@skipIfXpu / subtest(..., decorators=[skipIfXpu(...)]) mirroring the skip
cells from third_party/torch-xpu-ops/test/xpu/skip_list_common.py. Each skip
cites the matching intel/torch-xpu-ops tracking issue.

Authored by Claude.
```

- **NEVER** add a `Co-authored-by:` trailer for the AI assistant (interferes with the Linux Foundation CLA bot — see `AGENTS.md`).

If the porting is purely single-axis (only generalization, or only XPU enable on an already-generalized class), one commit is fine. The two-phase split is required only when both axes are touched on the same branch.

### Step 4: Search Related Files for Patterns

Before finalizing changes, search for similar patterns in nearby files:
```bash
# Find related test files with same patterns
grep -l "fuser1\|cuda.is_available\|instantiate_device_type_tests" test/directory/
```

Search for existing accelerator usage patterns:
```bash
grep -rn "accelerator\." test/ | head -20
```

### Step 5: Final Local Verification

By this point you have already run two per-phase test gates:

1. **Phase 1 gate** (inside Step 3a) — CUDA-only suite, proves the refactor is a no-op.
2. **Phase 2 gate** (inside Step 3b) — CUDA + XPU suites, proves XPU enablement works and CUDA is still green.

Step 5 is the **final** pre-push aggregation: run the affected file end-to-end one more time from the clean post-Phase-2 tree, with no `-k` filter, to catch anything the targeted runs missed (e.g. helper tests outside the class, module-level imports).

Use nightly PyTorch wheel with XPU support:
```bash
source ~/miniforge3/bin/activate pytorch_opencode_env
cd /tmp
python -m pytest /home/daisydeng/daisy_pytorch/test/path/to/test_file.py -v -k "TestClassName"
```

Run specific XPU tests:
```bash
source ~/miniforge3/bin/activate pytorch_opencode_env
cd /tmp
python -m pytest /home/daisydeng/daisy_pytorch/test/path/to/test_file.py::TestClassName::test_name -v
```

Verify API compatibility:
```bash
cd /tmp && source ~/miniforge3/bin/activate pytorch_opencode_env
python -c "from torch import accelerator; print(accelerator.current_accelerator())"
```

For local validation, prefer running from `/tmp` with the exact target XPU classes in the changed file. Example:

```bash
source ~/miniforge3/bin/activate pytorch_opencode_env
cd /tmp
python -m pytest /home/daisyden/upstream/upstream_ut/test/complex_tensor/test_complex_tensor.py -v -k "TestComplexBwdGradientsXPU or TestComplexTensorXPU"
```

All targeted XPU tests should pass (or skip with a tracking-issue reason) on the local validation environment before claiming completion.

If verification fails:

1. Check whether the failure is from the test enablement change or a pre-existing/backend issue.
2. If it is fixable in the test change, fix it and rerun.
3. If it still fails and needs issue filing, load `unittest_dev/submit_ut_issues` and prepare the issue details.
4. Ask the user for confirmation before submitting any issue. Never auto-submit.

### Step 6: Verify Commit Hygiene Before Push

By this point you have two atomic commits from Step 3a and Step 3b (or a single commit if the work was single-axis). No restructure is needed when commits are made incrementally — that is the whole point of the Step 3a/3b split.

Verify before push:

```bash
# 1. Phase 1 commit (HEAD~1) must contain ZERO xpu tokens
git show HEAD~1 -- test/<file>.py | grep -i xpu
# MUST return zero matches.

# 2. Phase 1 commit must keep only_for=("cuda",) unchanged
git show HEAD~1 -- test/<file>.py | grep 'only_for='
# MUST show only_for=("cuda",) — NOT ("cuda", "xpu").

# 3. Phase 2 commit must contain the XPU enablement
git show HEAD -- test/<file>.py | grep -E 'only_for=\("cuda", "xpu"\)|allow_xpu=True|skipIfXpu'
# MUST show at least one match.

# 4. Every @skipIfXpu must cite a tracking issue
git show HEAD -- test/<file>.py | grep -B0 -A0 'skipIfXpu' | grep -v 'intel/torch-xpu-ops#'
# MUST be empty (every skipIfXpu line has an accompanying tracking-issue reference).
```

If any check fails, fix it as an additional commit on top (or amend the offending commit if it has not been pushed yet) — see the [Recovery: Restructure Messy History](#recovery-restructure-messy-history) appendix only if the commits are tangled beyond simple amend.

### Recovery: Restructure Messy History

**Use only when Step 3a/3b discipline was not followed and the branch already has several mixed WIP commits.** If you committed incrementally as Step 3 instructs, skip this section entirely.

```bash
# 1. Safety net (NEVER skip this)
git branch backup-pre-restructure HEAD

# 2. Reset to base and rebuild as two clean commits
git reset --hard <BASE_COMMIT>
# ... re-apply Phase 1 edits, commit ...
# ... re-apply Phase 2 edits, commit ...

# 3. Verify the restructured tree is BYTE-IDENTICAL to the original HEAD
diff <(git show backup-pre-restructure:test/<file>.py) \
     <(git show HEAD:test/<file>.py)
# MUST produce zero output. If anything differs, restructure is wrong — do NOT push.

# 4. Re-run the Step 6 verification grep checks.
```

If verification fails: `git reset --hard backup-pre-restructure` restores the pre-restructure state losslessly. Do NOT push a restructured branch whose tree differs from the verified pre-restructure HEAD.

### Step 7: Submit GitHub PR (regular fork PR — NOT ghstack)

Push to your **personal fork** of `pytorch/pytorch` and open a normal draft PR. Do **NOT** use ghstack for this workflow: ghstack splits each commit into a separate PR, which fragments review of what is logically one change. A single PR with two well-labelled commits is the right granularity.

#### 7.1: Push to Fork with `--force-with-lease`

```bash
# Assumes remotes:
#   origin   -> pytorch/pytorch
#   <fork>   -> <your-user>/pytorch    (e.g. daisyden)
git push -u <fork> <branch> --force-with-lease
```

**Force-push rules:**
- ALWAYS use `--force-with-lease`, NEVER `--force`.
- If `--force-with-lease` rejects, fetch and reconcile — do NOT escalate to `--force`.
- NEVER force-push to `main`/`master` on any remote.

#### 7.2: Prepare PR Summary
Display the following to user for confirmation:

**PR Title:** `Generalize <ClassName> for XPU and other accelerators`

**Two-commit structure:**
- Commit 1: rename + `device_mod` refactor; class still `only_for=("cuda",)`; pure no-op on CUDA.
- Commit 2: `only_for=("cuda", "xpu"), allow_xpu=True`; XPUGraph plumbing; `@skipIfXpu` mirroring `skip_list_common.py` with tracking-issue citations.

**Test Verification:**
- CUDA: existing variants pass unchanged after Commit 1 alone.
- XPU: same variants run after Commit 2; cells in `skip_list_common.py` are skipped with reason strings linking to torch-xpu-ops issues.

**Target base:** `pytorch/pytorch:main`

#### 7.3: Wait for User Confirmation
Use `question` tool to ask user for confirmation before proceeding.

#### 7.4: Create Draft PR After Confirmation

```bash
gh pr create \
  --repo pytorch/pytorch \
  --base main \
  --head <your-user>:<branch> \
  --draft \
  --title "Generalize <ClassName> for XPU and other accelerators" \
  --body "$(cat <<'EOF'
## Summary
- Commit 1: rename `TestSDPACudaOnly` -> `TestSDPAGpuOnly` and replace hardcoded
  `torch.cuda.*` with `device_mod = getattr(torch, torch.device(device).type)`.
  Pure CUDA behavior unchanged; class still instantiated `only_for=("cuda",)`.
- Commit 2: enable XPU via `only_for=("cuda", "xpu"), allow_xpu=True`, add
  XPUGraph plumbing where applicable, and mirror skip cells from
  `third_party/torch-xpu-ops/test/xpu/skip_list_common.py` using `@skipIfXpu`
  and `subtest(..., decorators=[skipIfXpu(...)])`. Every skip cites an
  `intel/torch-xpu-ops#NNNN` tracking issue.

## Test plan
- CUDA: existing parametrize variants pass unchanged.
- XPU: same variants run; cells listed in `skip_list_common.py` are skipped
  with reason strings linking to torch-xpu-ops tracking issues.

Authored by Claude.
EOF
)"
```

Return the PR URL to user.

**NEVER auto-submit without user confirmation.**
**NEVER use ghstack for this workflow** — regular fork PR only.

## Key Patterns Learned from PyTorch PRs

From analyzing PRs #178849, #179549, #176689, #176688, #178565, #166396, #174058, #174057, #174056, #174054, #174053:

1. **Device availability check**: `torch.cuda.is_available() or torch.xpu.is_available()`
2. **Hardcoded device strings**: Change to accelerator-aware runtime selection
3. **Test instantiation**: `instantiate_device_type_tests(allow_xpu=True)`
4. **DecorateInfo device_type**: Use `device_type=("cuda", "xpu")` for skipping/decorating tests

## torch.accelerator API Reference (Verified)

All methods verified in PyTorch nightly (`pytorch_opencode_env`):

| Method | Returns | Description |
|--------|---------|-------------|
| `accelerator.is_available()` | `bool` | Check if accelerator is available |
| `accelerator.current_accelerator()` | `str` | Get current accelerator name ("xpu", "cuda", etc.) |
| `accelerator.device_count()` | `int` | Number of available devices |
| `accelerator.current_device_index()` | `int` | Get current device index |
| `accelerator.set_device_index(idx)` | `None` | Set current device index |
| `accelerator.set_device_idx(idx)` | `None` | Alias for set_device_index |
| `accelerator.get_device_capability(device)` | `dict` | Get device compute capability |
| `accelerator.memory_allocated(device)` | `int` | Get bytes allocated on device |
| `accelerator.max_memory_allocated(device)` | `int` | Get max bytes allocated |
| `accelerator.synchronize(device)` | `None` | Synchronize device |
| `accelerator.current_stream(device)` | `torch.Stream` | Get current stream |
| `accelerator.set_stream(stream)` | `None` | Set current stream |
| `accelerator.empty_cache()` | `None` | Empty accelerator cache |
| `accelerator.device_index(device)` | `None` | Set device by index |

## Backend Device String Mapping

| Backend | Device String | torch.accelerator Name |
|---------|---------------|------------------------|
| NVIDIA GPU | "cuda" | "cuda" |
| Intel GPU | "xpu" | "xpu" |
| Apple Silicon | "mps" | "mps" |
| Custom | "privateuseone" | "privateuseone" |

## Constraints

- **CUDA IPC tests**: Do NOT port - XPU has no IPC equivalent
  - `test_rebuild_cuda_tensor`, IPC handle functions
  - Any test using `_share_cuda_()`, `ipc_collect()`
  - CUDA caching allocator IPC mechanisms

- **CPU tests**: Skip - already device-agnostic, no XPU-specific handling needed

- **Build requirement**: No local build needed - use pytorch_opencode_env nightly wheel

- **Test verification**: Run tests from `/tmp` to avoid local pytorch shadowing conda env

- **Local validation env**: Use `source ~/miniforge3/bin/activate pytorch_opencode_env`

- **Issue submission**: If local XPU verification still fails after test-side fixes, use `unittest_dev/submit_ut_issues` to prepare submission, but always ask the user before creating the issue

- **PR confirmation**: Always confirm PR details with user before submission

- **API verification**: Always verify APIs against actual PyTorch documentation

## File Assessment Checklist

When evaluating a test file for XPU porting:

- [ ] Check test doesn't already exist in torch-xpu-ops/test/xpu/
- [ ] Distinguish GPU tests from CPU tests (only port GPU tests)
- [ ] File has CUDA-specific tests (@skipIf(not TEST_CUDA_IPC))
- [ ] Check for Event/Stream equivalents (torch.xpu.Event/Stream)
- [ ] File uses torch.cuda.ipc_* APIs (may need to skip entire file)
- [ ] File uses instantiate_device_type_tests with allow_xpu
- [ ] File has hardcoded device="cuda" strings in GPU paths
- [ ] File has CUDA-dependent helper functions needing generalization

### Pre-Push Checklist (two-commit split)

When the work landed as Commit 1 (generalization) + Commit 2 (XPU enable + skips) — made incrementally per Step 3a/3b:

- [ ] Commit 1 contains **zero** `xpu` tokens: `git show HEAD~1 -- test/<file>.py | grep -i xpu` returns nothing
- [ ] Commit 1 keeps `instantiate_device_type_tests(..., only_for=("cuda",))` unchanged
- [ ] **Phase 1 local-test gate passed** in `pytorch_opencode_env`: renamed class runs with zero failures and zero collection errors; CUDA-only cells skip cleanly (expected on the XPU host)
- [ ] Commit 2 contains the `only_for=("cuda", "xpu")` extension, `allow_xpu=True`, XPUGraph plumbing (if any), and ALL `@skipIfXpu` / `subtest(..., decorators=[skipIfXpu(...)])` / in-body `self.skipTest("xpu", ...)` calls
- [ ] **Phase 2 local-test gate passed** in `pytorch_opencode_env`: at least one XPU row is exercised (PASSED or SKIPPED with tracking-issue reason); no new failures vs Phase 1; CUDA-cell skip set unchanged
- [ ] Every XPU skip cites an `intel/torch-xpu-ops#NNNN` tracking issue — no bare `@skipIfXpu`
- [ ] Skip-list parity: every cell skipped by `third_party/torch-xpu-ops/test/xpu/skip_list_common.py` for this class is mirrored in-place, and vice versa (no double-skip)
- [ ] **Final Step 5 verification passed**: full-file run (no `-k` filter) on the post-Phase-2 tree
- [ ] Pushed to **personal fork** with `--force-with-lease` (NEVER `--force`)
- [ ] Draft PR opened against `pytorch/pytorch:main` as a regular fork PR (NOT ghstack); PR body explains the two-commit split
- [ ] Commit messages include `Authored by Claude.` trailer; NO `Co-authored-by:` AI trailer (CLA bot)

If the branch instead came from the [Recovery: Restructure Messy History](#recovery-restructure-messy-history) path, ALSO verify:

- [ ] `backup-pre-restructure` branch exists and was created before the reset
- [ ] `diff <(git show backup-pre-restructure:test/<file>.py) <(git show HEAD:test/<file>.py)` produces zero output

## Testing Workflow

### Verify accelerator API
```bash
cd /tmp && source ~/miniforge3/bin/activate pytorch_opencode_env
python -c "from torch import accelerator; print(accelerator.current_accelerator())"
```

### Test Instantiation Pattern
Check if file uses parametrized device tests:
```bash
grep "instantiate_device_type_tests" test_file.py
```

If present, ensure `allow_xpu=True`:
```python
instantiate_device_type_tests(TestClass, globals(), allow_xpu=True)
```

### Direct Device Tests
For tests with explicit device in test method:
```python
def test_gpu_function(self):
    # Use accelerator API for device-agnostic code
    if accelerator.is_available():
        device = accelerator.current_accelerator()
    else:
        device = "cpu"
    t = torch.randn(3, 3, device=device)
    # test logic
```

### Local Verification Gate

Before declaring the port complete:

1. Activate `pytorch_opencode_env` with `source ~/miniforge3/bin/activate pytorch_opencode_env`.
2. Run the changed test file from `/tmp` against the repo path.
3. Prefer a focused `-k` expression covering the newly enabled XPU classes first.
4. If failures remain after reasonable test-side fixes, switch to `unittest_dev/submit_ut_issues` for issue preparation.
5. Present the failure summary to the user and ask before submitting any issue.

## Examples

### test_memory_efficient_fusion.py (Ported)
- Uses TorchScript fuser "fuser1" which supports XPU
- Updated HAS_GPU constant and device selection logic
- 7 tests run on XPU with runtime device selection
- Success: All tests pass

### test_multiprocessing.py (Skipped)
- CUDA IPC-specific mechanisms
- No XPU equivalent for torch.cuda.ipc_* APIs
- Conclusion: Not portable - skip entire file

## Best Practices

1. **Check precondition first** - verify test doesn't exist in torch-xpu-ops
2. **Prioritize torch.accelerator API** for cross-backend device-agnostic code
3. **Verify all APIs** against actual PyTorch documentation before documenting
4. **Distinguish GPU vs CPU tests** - only port GPU-specific tests
5. **Check XPU availability first** when generalizing device selection
6. **Use runtime device selection** over hardcoded device strings
7. **Remove stale comments** after updates
8. **Test from correct directory** to avoid module shadowing
9. **Document non-portable tests** with clear explanation
10. **Always get PR confirmation** before submitting to GitHub
11. **Verify ALL CUDA patterns** are addressed before completion

## Common Patterns Found

| CUDA Pattern | XPU Portability | Preferred Solution |
|--------------|------------------|---------------------|
| `torch.cuda.is_available()` | `accelerator.is_available()` | torch.accelerator API |
| `torch.cuda.current_device()` | `accelerator.current_device_index()` | torch.accelerator API |
| `torch.cuda.Event()` | `torch.xpu.Event()` | Equivalent available |
| `torch.cuda.Stream()` | `torch.xpu.Stream()` | Equivalent available |
| `torch.cuda.ipc_collect()` | NOT portable | No XPU IPC - skip |
| `_share_cuda_()` | NOT portable | No XPU IPC - skip |
| `device="cuda"` | Runtime selection | Current accelerator selection or `device=device` |
| `.cuda()` method | `.to(accelerator.current_accelerator())` | Device selection |
| `@onlyCUDA` | `@onlyAccelerator` | Decorator update |
| `@onlyOn(["cuda"])` | `@onlyAccelerator` | Decorator update |
