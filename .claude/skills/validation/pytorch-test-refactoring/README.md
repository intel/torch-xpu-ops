# PyTorch Test Refactoring Workflow

This directory contains skills for decoupling PyTorch tests from specific hardware accelerators. The goal is to make tests run on any accelerator (CUDA, XPU, MPS, etc.) without modification, while preserving coverage for truly device-specific features.

## Overview

```
┌──────────────────────────────────────────────────────────┐
│                   Full Workflow                           │
│                                                          │
│   1. CLASSIFY ──► 2. REFACTOR ──► 3. REVIEW ──► 4. PR   │
│      (classify-    (refactor-       (review-test-  (submit-│
│       test-files)   test-            refactoring)   refactoring-│
│                     decoupling)                    pr)   │
└──────────────────────────────────────────────────────────┘
```

## Skill Inventory

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| **`refactor-test-decoupling`** | Refactor a test file into S1/S2/S3 classes | You have a test file to decouple |
| **`review-test-refactoring`** | Audit a refactored file for correctness | Before submitting a PR, or when reviewing someone else's |
| **`classify-test-files`** | Scan & classify test files by device dependency | You need to triage which files need refactoring |
| **`submit-refactoring-pr`** | Create a PR with verified refactoring | Refactoring is done and verified |
| **`ci-automation`** | Debug CI failures on refactoring PRs | CI is failing on a refactoring PR |
| **`gen_test_class_table`** | Generate the merged classification spreadsheet | You need the full classification table |
| **`case-summary`** | Annotate classification with test case counts | You need weighted coverage metrics |

## The Two Core Skills

### `refactor-test-decoupling`

**What it does:** Transforms a PyTorch test file by splitting tests into three categories:

- **S1 — Accelerator-Unrelated**: CPU-only tests. Class name: `TestFoo`. Plain `TestCase` or `@instantiate_parametrized_tests`.
- **S2 — Accelerator-Agnostic**: Tests that use a device but only generic APIs. Class name: `TestFooDevice`. Uses `instantiate_device_type_tests()`.
- **S3 — Accelerator-Specific**: Tests requiring a particular accelerator's unique features. Class name: `TestFooCUDA`/`TestFooXPU`/etc. Uses `instantiate_device_type_tests(..., only_for=...)` or plain `TestCase` with guard.

**How to invoke:**

```
# Via an agent:
Invoke the refactor-test-decoupling skill with the target file.

# Expected input:
test/<path>/<test_file.py>   TestClass   conda_env=<env>   pytorch_folder=<path>
```

**The agent will:**
1. Read the file and classify every test method using the decision tree
2. Create new S1/S2/S3 class(es) following naming conventions
3. Replace `@onlyCUDA` → `@onlyAccelerator`, `device="cuda"` → `device` param, etc.
4. Fix imports and instantiation mechanisms
5. Check for external references (`dynamo_skips/`, `common_methods_invocations.py`, `slow_tests.json`)
6. Verify test count and syntax

**Classification decision tree:**

```
Does the test reference a device?
├─ NO → S1
├─ YES → What device APIs?
│  ├─ Generic only (torch.device(device), make_tensor(..., device=device)) → S2
│  ├─ Category A or B APIs (empty_cache, Stream, etc.) → S2
│  ├─ Category C APIs (NCCL, NVTX, cuDNN, etc.) → S3
│  └─ Hard to tell → Leave as-is
```

**False-CUDA patterns** (look like CUDA, but are S2):

| Pattern | Action |
|---------|--------|
| `@onlyCUDA` on standard ops | Replace with `@onlyAccelerator` |
| `.cuda()` / `device="cuda"` | Replace with `.to(device)` / `device` param |
| `test_foo_cuda` naming | Remove `_cuda` suffix |
| `@skipIf(not TEST_CUDA, ...)` | Replace with `@onlyAccelerator` |

**Key rule:** `@onlyAccelerator` is a **method decorator**, never a class decorator.

---

### `review-test-refactoring`

**What it does:** Audits a refactored test file (or a PR/diff) against the decoupling standards. It checks classification correctness, naming, API replacements, instantiation, imports, and completeness.

**How to invoke:**

```
# Whole-file review (pass a test file path):
Invoke the review-test-refactoring skill with a file path.

# Diff-based review (pass a PR URL, branch name, or git diff):
Invoke the review-test-refactoring skill with the diff source.
```

**What the review checks:**

| Check | Severity if wrong |
|-------|-------------------|
| S3 test that should be S2 (false-CUDA) | **Blocker** — locks tests out of other accelerators |
| S2 test that should be S3 (uses Cat C APIs) | **Blocker** — will fail on other accelerators |
| Class naming wrong (e.g., `TestFoo` for S2) | **Major** |
| Wrong instantiation mechanism | **Blocker** — class won't run |
| Stale imports (`TEST_CUDA` no longer needed) | **Major** |
| `@onlyAccelerator` used as class decorator | **Blocker** — breaks instantiation |
| Stale `dynamo_skips/` or `DecorateInfo` references | **Blocker** — tests silently skip or fail |

**Review output format:**

```
## Review: <file>

### Summary
- Classification: correct / N issues
- Naming: correct / N issues
- API replacements: correct / N issues

### Blockers (must fix)
### Major (should fix)
### Minor (nice to have)
```

---

## Recommended Workflow

### Step 1 — Audit

Use `classify-test-files` to identify which files need refactoring, or use `gen_test_class_table` to get the full classification spreadsheet.

### Step 2 — Refactor

For each target file, invoke `refactor-test-decoupling`. The skill reads the file, classifies tests, and rewrites it.

**During refactoring, the agent handles these automatically:**
- [x] Classifies every test method (S1/S2/S3)
- [x] Creates correctly-named classes
- [x] Replaces device-specific APIs with generic equivalents
- [x] Updates imports and instantiation
- [x] Scans `common_methods_invocations.py` for stale DecorateInfo entries
- [x] Finds stale `dynamo_skips/` / `dynamo_expected_failures/` entries
- [x] Verifies test count is preserved

### Step 3 — Review

Before submitting, run `review-test-refactoring` on the refactored file. This is your quality gate — it catches:

- Tests that should have been S2 but were left as S3 (most common error)
- Tests that should have been S3 but were generalized to S2
- Wrong naming, wrong instantiation, broken references
- Missing imports or stale imports

**Always review before submitting.** This catches issues that the refactoring skill might miss (especially around edge-case API classifications and dtype compatibility on MPS).

### Step 4 — Submit

Use `submit-refactoring-pr` to create a PR. This skill takes the verified refactoring and packages it into a draft PR against `pytorch/pytorch`.

```
refactor-test-decoupling → review-test-refactoring → submit-refactoring-pr
```

**What it does:**
1. Inspects the working tree — confirms only intended test files changed
2. Collects test count evidence (`grep -c 'def test_'` before/after)
3. Creates a feature branch (e.g. `refactor/test_dataloader`)
4. Rebases onto upstream `pytorch/pytorch` `viable/strict` (avoids release-vs-main diff pollution)
5. Stages only the refactored test files (explicit paths, never `git add -A`)
6. Drafts a commit with test count evidence in the body
7. **Confirm-gated** — presents everything to you for approval before pushing
8. On approval: commits, pushes to `daisyden/pytorch` fork with `--force-with-lease`, opens a draft PR via `gh`

**Key constraints:**
- **Confirm-gated**: never pushes or creates a PR without explicit user approval
- **Regular fork PR** — NOT ghstack (head=`daisyden:<branch>`, base=`pytorch/pytorch:viable/strict`)
- **No logic changes**: the skill will reject a diff containing non-refactoring edits
- **Test count evidence required** in both commit body and PR description
- `git add` uses explicit paths only — never `-A` blindly

**How to invoke:**

```
# Via an agent:
Invoke the submit-refactoring-pr skill.
Requires: <pytorch_folder> and user confirmation before push.
```

---

## Real-World Examples

### PR #189250 — Decouple `test_dataloader` and `test_comparison_utils`

[`pytorch/pytorch#189250`](https://github.com/pytorch/pytorch/pull/189250)

A full S1/S2/S3 refactoring across two files, demonstrating all three strategies and the class rename workflow.

**File 1: `test_dataloader.py` (148 additions, 121 deletions)**

| Class | Strategy | Tests | Rationale |
|-------|----------|-------|-----------|
| `TestDataLoader` | **S1** | 66 | CPU-only tests, no device usage |
| `TestDataLoaderDevice` *(renamed from `TestDataLoaderDeviceType`)* | **S2** | pin_memory tests | Uses generic device APIs, `instantiate_device_type_tests()` |
| `TestDictDataLoaderDevice` | **S2** | pin_memory tests | Same pattern with accelerator-generic guards |
| `TestDataLoaderCUDA` | **S3** | 4 CUDA IPC tests | Uses `torch.cuda` / `TEST_CUDA_IPC` — Category C APIs only |

**S2 rename applied:** `TestDataLoaderDeviceType` → `TestDataLoaderDevice` per convention.

**File 2: `test_comparison_utils.py` (15 additions, 9 deletions)**

| Class | Strategy | Tests | Rationale |
|-------|----------|-------|-----------|
| `TestComparisonUtils` | **S1** | 6 | CPU-only tests |
| `TestComparisonUtilsDevice` | **S2** | 1 (`test_assert_device`) | Converted `@unittest.skipIf(not torch.cuda.is_available)` → `@onlyAccelerator` + `instantiate_device_type_tests()` |

**Workflow demonstrated:**
1. Audit: classify all test methods per decision tree
2. Split: create S1/S2/S3 classes with correct naming
3. Enlarge whitelist: `@onlyCUDA` → `@onlyAccelerator`
4. Preserve blacklist: `@skipIfXpu`, `@skipIfRocm` kept as-is
5. Verify: test count preserved (`126 → 126`, `7 → 7`)

---

## Quick Reference: Common Pitfalls

| Pitfall | How to Avoid |
|---------|-------------|
| `@onlyAccelerator` on a class | Always use as **method decorator** |
| S1 class named `TestFooCPU` | S1 = no device suffix. Use `TestFoo`. |
| S2 class named `TestFoo` | S2 = `TestFooDevice` suffix |
| `skipIfXpu` from `common_utils` in S2 class | Use `common_device_type` equivalents (`skipXPUIf`, `skipCUDAIf`) |
| `instantiate_device_type_tests` for S1 | Creates wasteful per-device variants |
| Forgetting to check `dynamo_skips/` after rename | Stale entries cause silent CI failures |
| Forgetting to update `DecorateInfo` after rename | Previously-skipped tests start running and failing |
| Category A/B APIs treated as CUDA-specific | Check `reference/device_api_catalog.yaml` |

## Reference Material

- `reference/device_api_catalog.yaml` — Classifies every device API as Category A (accelerator-equivalent), B (general concept), or C (device-specific). **Always consult this for classification decisions.**
- `examples/` — Worked examples of test file refactoring (see `TEST_BENCHMARK_UTILS_REFACTORING.md` for a walkthrough).
