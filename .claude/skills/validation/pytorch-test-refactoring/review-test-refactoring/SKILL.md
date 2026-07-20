---
name: review-test-refactoring
description: >
  Review PyTorch test refactoring for correctness and completeness against the
  decoupling standards defined in the refactor-test-decoupling skill. Accepts a
  test file path (whole-file review), a PR URL, a git diff, or a branch name.
  Use this when asked to review a test refactoring PR, check a test decoupling
  change, verify a refactored test file, review a single test file for
  classification correctness, or when the user mentions "review" in the context
  of test splitting, test decoupling, test refactoring, or accelerator-agnostic
  test migration. Also use when the user opens a PR, diff, or Python test file
  and asks for a quality check.
---

# Review Test Refactoring

Review a test file (or a PR/diff) against the decoupling standards defined in
the `refactor-test-decoupling` skill. The review checks classification
correctness, naming conventions, API replacements, instantiation mechanisms,
and completeness.

## How to Use This Skill

### Choose Your Mode

This skill supports two review modes. Determine which applies:

| Input | Mode | What to Review |
|-------|------|----------------|
| A test file path (`test/test_ops.py`) | **Whole-file review** | Audit the entire file against decoupling standards |
| A PR URL, branch name, or `git diff` | **Diff-based review** | Review only the changed portions |

### For Whole-File Review (test file path)

1. **Read the entire test file.** Load the full file — you are auditing every
   class and test method, not just a diff.

2. **Read `../reference/device_api_catalog.yaml`.** This catalog is the
   authoritative reference for whether a device API is Category A (has
   `torch.accelerator` equivalent), Category B (general cross-backend concept),
   or Category C (truly device-specific). Every API classification decision in
   the review must be grounded in this catalog. See
   `../reference/classification_guide.md` for lookup instructions.

3. Run through the full checklist below, applying every check to every class
   and test method in the file. There is no "before" version to compare against
   — you are the auditor.

### For Diff-Based Review (PR, branch, or diff)

1. Identify the changes to review. If the user provides a PR URL, fetch it with
   `gh`. Otherwise, use `git diff` against the base branch (usually `origin/main`).

2. **Read `../reference/device_api_catalog.yaml`** (same as above).

3. For each changed test file, run through the checklist below. Focus on the
   diff — you are reviewing what changed, not re-auditing the entire file.
   However, key checks like naming conventions, instantiation mechanisms, and
   import cleanliness should be verified for the file as a whole even in
   diff-based mode.

### Reporting

Report findings organized by severity:
- **Blocker**: Test loss, wrong classification locking tests out of accelerators,
  broken instantiation (class won't run).
- **Major**: Wrong naming convention, wrong instantiation mechanism,
  stale imports that keep file classified as device_specific.
- **Minor**: Style issues, missed cleanup opportunities, suboptimal
  decorator ordering.

## Reference: The Device API Catalog

`../reference/device_api_catalog.yaml` classifies every PyTorch device API into three categories. Always consult it when reviewing classifications — never rely on memory or heuristics.

| Category | Description | Strategy Implication |
|----------|-------------|---------------------|
| **A** | APIs with `torch.accelerator` equivalents | **NOT device-specific** → Strategy 2 |
| **B** | General cross-backend concepts, no wrapper yet | **NOT device-specific** → Strategy 2 |
| **C** | Truly device-specific, no cross-device equivalent | **Strategy 3 only** |

**Rule**: Only Category C APIs justify Strategy 3. If a test uses only Category A or B APIs, it must be Strategy 2 with `@onlyAccelerator`.

## Review Checklist

### 1. Classification Correctness

The single most impactful category of review finding. A wrong classification
either locks tests out of accelerators they could run on (Strategy 2
misclassified as Strategy 3) or causes test failures on accelerators that lack
the required features (Strategy 3 misclassified as Strategy 2).

#### 1a. False-CUDA Detection (most common error)

For every test classified as Strategy 3 (`TestFooCUDA`) or using
`@onlyCUDA` / `device="cuda"`, ask:

> Is this test verifying a truly device-specific feature (Category C in the
> report), or is it just using CUDA as a device for generic computation?

**How to check**: Look up each `torch.cuda.*` API the test uses in
`../reference/device_api_catalog.yaml`. If every API it uses is Category A or B,
the test is misclassified — it should be Strategy 2 with `@onlyAccelerator`.

**Red flags** (signals the test is wrongly classified as CUDA-specific):

| Code Pattern | What It Means | Severity |
|-------------|---------------|----------|
| Test with generic ops (add, softmax, matmul, loss) still has `@onlyCUDA` or `device="cuda"` | Should be Strategy 2 with `@onlyAccelerator` | Blocker |
| `.cuda()` / `.to("cuda")` used instead of `.to(device)` | Test hardcodes CUDA for no reason | Blocker |
| `torch.cuda.<api>` call where the catalog shows `torch.accelerator.<api>` exists | Category A — has cross-accelerator equivalent; replace with `torch.accelerator.*` | Major |
| `torch.cuda.Stream` / `torch.cuda.Event` used but test not marked as Strategy 3 | Category B — general concept; verify usage context, usually Strategy 2 | Info |
| `TEST_CUDA` import remains but no Strategy 3 CUDA tests exist in the file | Stale import keeps file classified as device_specific | Major |

#### 1b. Over-generalization Detection

Conversely, check that tests using Category C APIs were NOT incorrectly
generalized to Strategy 2. Consult `../reference/device_api_catalog.yaml` → `category_c` for the full per-backend lists. Key examples:

| Code Pattern | What It Means | Severity |
|-------------|---------------|----------|
| Test using any API from `category_c.cuda` in the catalog but placed in `TestFooDevice` with `@onlyAccelerator` | Will fail on non-CUDA accelerators | Blocker |
| Test using any API from `category_c.mps` in the catalog but placed in `TestFooDevice` | Will fail on non-MPS accelerators | Blocker |
| Test using any API from `category_c.xpu` in the catalog but placed in `TestFooDevice` | Will fail on non-XPU accelerators | Blocker |

**Dtype compatibility on MPS**: Even when a test uses only generic ops (no
Category C APIs), it may still fail on non-CUDA accelerators if it uses dtypes
not supported by that backend. The most common case: `complex128` and `float64`
are unsupported on MPS. When a test is generalized to Strategy 2 (or already
uses `@onlyAccelerator`):

| Check | How to Verify |
|-------|---------------|
| `complex128` or `torch.complex128` used in `@dtypes` or as default dtype | MPS does not support double-precision complex. Add `@expectedFailureMPS` or use `@dtypesIfMPS` to exclude `complex128`. |
| `float64` or `torch.float64` used in `@dtypes` or as default dtype | MPS does not support float64. Add `@expectedFailureMPS` or use `@dtypesIfMPS` to exclude `float64`. |
| `torch.long` used with MPS convolution/indexing ops | MPS has limited int64 support in some ops. Check if `@onlyNativeDeviceTypes` or a skip is needed. |

**How to validate**: For every test using `@onlyAccelerator` or the `device`
parameter, verify every dtype it exercises (from `@dtypes`, `_default_dtype`, or
inline tensor creation) against known MPS dtype limitations. The failure
signature is: `"Cannot convert a MPS Tensor to float64 dtype"` or similar dtype
conversion errors on MPS. In diff-based mode, pay special attention to tests
where `@onlyCUDA` was removed.

#### 1c. Strategy 1 Correctness

For tests in a Strategy 1 class (`TestFoo` without device suffix):

| Check | What to Look For |
|-------|-----------------|
| No `device` parameter in method signature | `def test_foo(self)` not `def test_foo(self, device)` |
| No device-dependent decorators | No `@onlyCUDA`, `@onlyAccelerator`, `@skipCUDAIf`, etc. |
| No `.to(device)`, `.cuda()`, `device=device` in test body | All tensors are CPU |
| No cross-device tensor operations | Can't have CPU tensor op GPU tensor |

### 2. Naming Convention

Verify class names follow the convention from `refactor-test-decoupling`:

| Strategy | Expected Name | Wrong Name Examples |
|----------|--------------|---------------------|
| Strategy 1 (accelerator-unrelated) | `TestFoo` (original name, no device suffix) | `TestFooCPU` |
| Strategy 2 (accelerator-agnostic) | `TestFooDevice` | `TestFoo`, `TestFooGeneric` |
| Strategy 3 (accelerator-specific) | `TestFooCUDA`, `TestFooMPS`, `TestFooXPU` | `TestFooDeviceCUDA` |

#### 2a. Cross-File Reference Integrity

**When a class is renamed** (e.g., `TestIndexing` → `TestIndexingDevice`), the
old name may still be referenced in external configuration files. A rename
without updating these files causes CI breakage: dynamo expected-failure entries
stop matching, turning expected failures into unexpected failures.

**Files to check for stale class name references:**

| File/Directory | Example Reference | How to Verify |
|---------------|-------------------|---------------|
| `test/dynamo_skips/` | `TestIndexing.test_invalid_sparse_coo_values_cpu` | `find test/dynamo_skips/ -name "OldClassName*"` **(by filename, NOT grep — these are sentinel files, often 0 bytes)** |
| `test/dynamo_expected_failures/` | `TestIndexingCPU.test_byte_mask_cpu` | `find test/dynamo_expected_failures/ -name "OldClassName*"` **(by filename, NOT grep)** |
| `test/inductor_expected_failures/` | `TestIndexing.test_foo` | `find test/inductor_expected_failures/ -name "OldClassName*"` **(by filename, NOT grep)** |
| `torch/testing/_internal/common_methods_invocations.py` | `DecorateInfo(unittest.skip("..."), 'TestCommon', 'test_complex_half_reference_testing')` | Search for `'OldClassName'` string in `DecorateInfo(...)` constructor calls |
| `.ci/pytorch/test_exclude_list.py` | Test name in skip list | `grep -r "OldClassName\b" .ci/pytorch/` |
| `.ci/pytorch/*-trunk.yml` | Test name in CI config | `grep -r "OldClassName\b" .ci/` |

**How to fix:** For each stale reference, update the class name to match the
new name. Verify which class actually owns each test — when a class is split
into multiple new classes (Strategy 1 + 2 + 3), tests may now live under
different class names (e.g., `TestIndexing.test_foo` might now be
`TestIndexingDevice.test_foo` or `TestIndexingCPU.test_foo`).

**For `common_methods_invocations.py` specifically:** `DecorateInfo` entries
use exact class name matching in `is_active()` — if `cls_name='TestCommon'`
but the test now lives in `TestCommonDevice`, the skip/xfail decorator is
silently dropped. To find broken entries:

```bash
python -c "
import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import DecorateInfo

old_names = {'TestOldName1', 'TestOldName2'}  # fill in renamed classes
count = 0
for op in op_db:
    for d in op.decorators:
        if isinstance(d, DecorateInfo) and d.cls_name in old_names:
            print(f'{op.name}: cls_name={d.cls_name}, test_name={d.test_name}')
            count += 1
print(f'Total: {count} stale DecorateInfo entries')
"
```

Then replace the old class name string literal with the new one in
`torch/testing/_internal/common_methods_invocations.py`.

**Severity**: Blocker — broken DecorateInfo entries cause tests to silently
run when they should be skipped, or tests to be silently skipped when they
should run.

### 3. Instantiation Mechanism

| Strategy | Expected Mechanism | Wrong Mechanism |
|----------|-------------------|-----------------|
| Strategy 1, no parametrization | Plain `TestCase` | `instantiate_device_type_tests` |
| Strategy 1, with `@parametrize`/`@ops`/`@dtypes` | `@instantiate_parametrized_tests` | `instantiate_device_type_tests` |
| Strategy 2 | `instantiate_device_type_tests(TestFooDevice, globals())` | `@instantiate_parametrized_tests` |
| Strategy 3, no parametrization | Plain `TestCase` with `setUp` guard | `instantiate_device_type_tests` |
| Strategy 3, with parametrization | `@instantiate_parametrized_tests` | `instantiate_device_type_tests` |

**Critical**: Check that `instantiate_device_type_tests` is never used for
CPU-only (Strategy 1) classes — it creates useless per-device variants.

**Critical**: Check that no class uses both `instantiate_parametrized_tests`
and `instantiate_device_type_tests` — double instantiation causes test name
collisions.

### 4. API Replacement Correctness

For Strategy 2 tests, verify device-specific APIs were replaced with their
device-agnostic equivalents. **Consult `../../../reference/device_api_catalog.yaml` → `category_a` for the authoritative mapping.** The catalog defines every `torch.<device>.<api>` → `torch.accelerator.<api>` replacement.

**Key checks:**

| Before (Wrong) | After (Correct) | Check |
|---------------|-----------------|-------|
| `@onlyCUDA` | `@onlyAccelerator` | Not left as `@onlyCUDA` |
| `@unittest.skipIf(not TEST_CUDA, ...)` | `@onlyAccelerator` | Not left as skip |
| `device="cuda"` | `device` parameter | No hardcoded `"cuda"` |
| `.cuda()` / `.to("cuda")` | `.to(device)` | No `.cuda()` calls |

For any `torch.cuda.<api>()` call remaining in a Strategy 2 test, check the
catalog: if Category A, it should be `torch.accelerator.<api>()`. If Category B,
use the unified type (e.g., `torch.Stream` instead of `torch.cuda.Stream`). If
Category C, the test belongs in Strategy 3.

### 5. Import Cleanup

| Check | How to Verify |
|-------|---------------|
| `TEST_CUDA` import removed if no Strategy 3 CUDA tests remain | `grep "TEST_CUDA"` in the file |
| `TEST_MPS` import removed if no Strategy 3 MPS tests remain | `grep "TEST_MPS"` in the file |
| `TEST_XPU` import removed if no Strategy 3 XPU tests remain | `grep "TEST_XPU"` in the file |
| `@onlyCUDA` import removed if no Strategy 3 CUDA tests remain | `grep "onlyCUDA"` in the imports |
| `@onlyOn` import removed if all uses were replaced | `grep "onlyOn"` in the file |
| New imports are correct | `onlyAccelerator` from `common_device_type`, `torch.accelerator` if used |

### 6. Test Completeness

**Whole-file review**: Verify every test method in the file is properly placed in
an appropriate strategy class. No test should be in a class that doesn't match
its device dependency level (e.g., a test using only CPU ops should not be in a
`TestFooCUDA` class).

| Check | How to Verify |
|-------|---------------|
| Every `def test_` belongs to the correct strategy class | Cross-reference each test's API usage against the catalog and its enclosing class name |
| `setUp` guards present for Strategy 3 | `self.skipTest` or `@unittest.skipIf` for device availability |
| No test logic unintentionally modified | If reviewing a diff, compare test bodies against the base version. If whole-file, flag tests that appear incomplete or have empty bodies |

**Diff-based review**: Additionally verify that every original test method is
accounted for (count `def test_` in old vs new). A test "lost" in refactoring is
a regression.

### 7. Common Pitfalls

| Pitfall | Detection | Severity |
|---------|-----------|----------|
| `@onlyAccelerator` used as class decorator | `@onlyAccelerator\nclass TestFoo` — breaks `instantiate_device_type_tests` | Blocker |
| Strategy 1 class has `device` parameter | `def test_foo(self, device)` in `TestFoo` (no Device suffix) | Blocker |
| `skipIfXpu`/`skipIfCUDA` from `common_utils` in device-agnostic class | These skip ALL variants, not just the target device | Major |
| `GPU_TYPE`/`HAS_GPU` from `inductor_utils` not converted | Leftover inductor-specific device abstraction | Major |
| Mixed `device` param and hardcoded `"cuda"` in same class | Inconsistent; some tests use device param, others hardcode | Major |
| `instantiate_device_type_tests` call references wrong class name | Class name in `globals()` call doesn't match actual test class — class never instantiated | Blocker |
| `except_for`/`only_for`/`allow_mps`/`allow_xpu` args missing from instantiation | Device allowlists not applied to current `instantiate_device_type_tests` call | Major |
| Category A/B API treated as if it makes a test CUDA-specific | Test locked to CUDA unnecessarily; check the catalog | Major |
| Missing blacklist skip decorators | `@skipXPU`, `@skipMPS`, `@skipMeta` absent — these document known gaps. If the original file had them and they're now gone, that's a regression | Blocker |
| `@onlyAccelerator` used without dtype compatibility check | Test runs on MPS/XPU but uses `complex128` or `float64` (unsupported on MPS). For every test using `@onlyAccelerator`, verify every dtype the test uses is supported on ALL target backends. If not, add `@expectedFailureMPS`, `@dtypesIfMPS`, or a skip decorator. | Blocker |
| Test class name doesn't match OpInfo DecorateInfo references | `DecorateInfo` entries in `common_methods_invocations.py` use exact class name matching in `is_active()`. If the test class in this file has a name that doesn't match existing `DecorateInfo` entries (check section 2a), previously-skipped tests will run and fail, or previously-xfailed tests will hard-fail. In whole-file review, verify the class names against `common_methods_invocations.py`. In diff-based review, check for renames. | Blocker |

### 8. Decorator Ordering

For Strategy 2 tests, decorators must be ordered correctly. The `device`
parameter is filled in by `instantiate_device_type_tests`, and other
parametrization decorators fill additional arguments:

```python
# Correct: @dtypes closest to method, @onlyAccelerator above
@onlyAccelerator         # outermost (skip if CPU)
@dtypes(torch.float32)   # parametrization
def test_foo(self, device, dtype):
    ...

# Wrong — @onlyAccelerator below @dtypes may cause issues
@dtypes(torch.float32)
@onlyAccelerator
def test_foo(self, device, dtype):  # incorrect
    ...
```

## Review Output Format

Structure your review as follows:

```
## Review: <test file path or PR/branch name>

### Summary
- Mode: whole-file / diff-based
- File(s) reviewed: N
- Classification: correct / N issues found
- Naming: correct / N issues found
- API replacements: correct / N issues found
- Completeness: all tests properly placed / N issues found

### Findings

#### Blockers (must fix)
- [ ] **<file:line>**: <issue description>
  - Fix: <suggested fix>

#### Major (should fix)
- [ ] **<file:line>**: <issue description>

#### Minor (nice to have)
- [ ] **<file:line>**: <issue description>

### Verified Correct
- <list of things that are correct per the standards>
```

## Reference

The refactoring standards this review checks against are defined in the
`refactor-test-decoupling` skill. Consult it for the full classification
decision tree, blacklist vs. whitelist rules, instantiation mechanism
comparison, and Strategy 1/2/3 patterns.

**`../reference/device_api_catalog.yaml`** is the single authoritative source for API classification. It categorizes every device API as A (accelerator equivalent), B (general concept), or C (truly device-specific). All classification decisions in the review must be grounded in this catalog — never hard-code or guess which APIs belong to which category.
