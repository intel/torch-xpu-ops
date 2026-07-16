---
name: refactor-test-decoupling
description: Refactor PyTorch test files to decouple tests from specific hardware accelerators using three strategies: accelerator-unrelated (CPU-only standalone classes with instantiate_parametrized_tests), accelerator-agnostic (device-generic classes with instantiate_device_type_tests), and accelerator-specific (single-accelerator standalone classes). Pay special attention to tests tagged @onlyCUDA or using .cuda()/device="cuda" that are NOT actually CUDA-specific — most should be refactored to accelerator-agnostic. Use when asked to decouple, refactor, or reorganize tests to work across multiple accelerators, or when a test file imports TEST_CUDA/TEST_MPS/TEST_XPU but most tests don't require a specific device.
---

# Refactor Test Decoupling

Refactor PyTorch test files so tests focus on core functional logic and are decoupled from specific hardware accelerators.

## Naming Convention

| Strategy | Class Name | Instantiation | Example |
|----------|-----------|---------------|---------|
| **Accelerator-unrelated** (S1) | `TestFoo` (original name) | `@instantiate_parametrized_tests` or plain `TestCase` | `TestBinaryUfuncs` |
| **Accelerator-agnostic** (S2) | `TestFooDevice` | `instantiate_device_type_tests()` | `TestBinaryUfuncsDevice` |
| **Accelerator-specific** (S3) | `TestFoo<Device>` | `instantiate_device_type_tests(TestFooCUDA, globals(), only_for="cuda")` when using `@dtypes`/`@dtypesIfCUDA`/`@dtypesIfCPU`; otherwise plain `TestCase` with `setUp` guard | `TestBinaryUfuncsCUDA` |

`instantiate_device_type_tests` **removes** the generic class from scope and replaces it with per-device variants (`TestFooDeviceCPU`, `TestFooDeviceCUDA`, etc.). `instantiate_parametrized_tests` keeps the class discoverable.

**S3 instantiation rule**: When an S3 class uses `@dtypes`, `@dtypesIfCUDA`, `@dtypesIfCPU`, or other device-type-aware decorators (which are designed for `instantiate_device_type_tests`), use `instantiate_device_type_tests(TestFooCUDA, globals(), only_for="cuda")` instead of `@instantiate_parametrized_tests`. Each test method receives `device` as its first parameter (always `"cuda"`), eliminating the need for per-method `@onlyCUDA` decorators or hardcoded `device = "cuda"` lines. This keeps mechanism consistency with S2 and lets `instantiate_device_type_tests` inject device-aware dtype resolution.

## Classification

Every test falls into one of three categories. Classification is hierarchical: **S3 > S2 > S1**.

| Category | Definition | Mechanism |
|----------|-----------|-----------|
| **Accelerator-unrelated (S1)** | No device usage; CPU only | `instantiate_parametrized_tests()` or plain `TestCase` |
| **Accelerator-agnostic (S2)** | Uses a device but only generic accelerator APIs | `instantiate_device_type_tests()` |
| **Accelerator-specific (S3)** | Requires a particular accelerator's unique features | `instantiate_device_type_tests(..., only_for=...)` when `@dtypes`-style decorators exist; otherwise plain `TestCase` |

### Device API Categories (consult `../reference/device_api_catalog.yaml`)

| Category | Examples | Strategy |
|----------|---------|----------|
| **A** — has `torch.accelerator` equivalent | `empty_cache`, `synchronize`, `CUDAGraph`, `memory_allocated`, `current_device` | S2 |
| **B** — general concept, no wrapper yet | `Stream`, `Event`, `manual_seed`, `get_device_properties` | S2 |
| **C** — truly device-specific, no cross-device equivalent | NCCL, NVTX, cuDNN, GDS, Jiterator, Metal shaders, SYCL handles | S3 |

**Only Category C makes a test S3.** If you can replace `"cuda"` with `"mps"` or `"xpu"` and the test still makes logical sense, it's S2.

### Blacklist vs. Whitelist Decorators

| Decorator Type | Examples | Principle | Action |
|---------------|----------|-----------|--------|
| **Blacklist** (explicit skips) | `@skipXPU`, `@skipCUDAIf`, `@skipMPS`, `@skipMeta`, `@onlyNativeDeviceTypesAnd` | Documents a **known gap** — intentional and informed | **KEEP as-is** |
| **Whitelist** (restrictive) | `@onlyCUDA`, `@onlyCPU`, `@onlyOn(["cuda","xpu"])`, `@unittest.skipIf(not TEST_CUDA, ...)` | Artificially **restricts** — usually historical accident | **ENLARGE** to `@onlyAccelerator` |

### Decision Tree

```
Does the test reference a device?
├─ NO → S1
├─ YES → What device APIs?
│  ├─ Generic only (torch.device(device), make_tensor(..., device=device)) → S2
│  ├─ Category A or B APIs → S2
│  ├─ Category C APIs → S3
│  └─ Hard to tell → Leave as-is

What decorators?
├─ Blacklist (@skipXPU, @skipCUDAIf, @skipMPS, @skipMeta, @onlyNativeDeviceTypesAnd) → KEEP
├─ Whitelist (@onlyCUDA, @onlyOn, @unittest.skipIf(not TEST_CUDA, ...)) → ENLARGE to @onlyAccelerator
```

### False-CUDA Patterns (→ S2, NOT S3)

These almost always indicate S2:

| Pattern | Why Not CUDA-Specific | Action |
|---------|----------------------|--------|
| `@onlyCUDA` on standard ops (add, softmax, matmul) | The op works on any accelerator | `@onlyAccelerator` + `device` param |
| `.cuda()` / `.to("cuda")` on tensors | Just device placement | `.to(device)` |
| `device="cuda"` in tensor creation | Any device would work | `device` param |
| `@unittest.skipIf(not TEST_CUDA, ...)` | Proxy for "needs accelerator" | `@onlyAccelerator` |
| Test name contains `_cuda` | Naming, not functional | Remove suffix |

## Strategy 1: Accelerator-Unrelated (S1)

Zero device dependency. CPU tensors only, no `device` parameter.

**Pattern A — Plain TestCase** (no parametrization):
```python
class TestFoo(TestCase):
    def test_basic_addition(self):
        a = torch.randn(3, 3)
        b = torch.randn(3, 3)
        self.assertEqual(a + b, torch.add(a, b))
```

**Pattern B — `@instantiate_parametrized_tests`** (has `@parametrize`/`@ops`/`@dtypes`):
```python
@instantiate_parametrized_tests
class TestFoo(TestCase):
    @parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_behavior(self, dtype):
        t = torch.randn(3, 3, dtype=dtype)
        self.assertEqual(t.softmax(0).sum(0), torch.ones(3, dtype=dtype))
```

**Why not `instantiate_device_type_tests`?** It creates per-device variants (TestFooCPU, TestFooCUDA, etc.) — wasteful when all variants do the same CPU-only work.

**Steps:**
1. Extract test methods into a standalone class named `TestFoo` (original name, no device suffix)
2. Remove `device` parameter from signatures; hardcode `"cpu"` or omit device args
3. Remove device decorators and device imports (`TEST_CUDA`, `TEST_MPS`, etc.)
4. Add `@instantiate_parametrized_tests` if the class has parametrized decorators

## Strategy 2: Accelerator-Agnostic (S2)

Tests that use a `device` parameter but only need generic accelerator APIs. **This is the highest-impact refactoring** — it unlocks tests for all accelerators at once.

### Canonical Before/After

**Before** (false-CUDA):
```python
from torch.testing._internal.common_cuda import TEST_CUDA

class TestFoo(TestCase):
    @unittest.skipIf(not TEST_CUDA, "no CUDA")
    def test_softmax_cuda(self):
        t = torch.randn(3, 3, device="cuda")
        result = t.softmax(0)
        self.assertEqual(result.sum(0), torch.ones(3, device="cuda"))

    @onlyCUDA
    @skipXPU  # XPU doesn't support this op yet
    def test_matmul_cuda(self, device):
        a = torch.randn(3, 3, device=device)
        b = torch.randn(3, 3, device=device)
        self.assertEqual(a @ b, torch.matmul(a, b))
```

**After** (accelerator-agnostic):
```python
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyAccelerator,
)

class TestFooDevice(TestCase):
    @onlyAccelerator
    def test_softmax(self, device):
        t = torch.randn(3, 3, device=device)
        result = t.softmax(0)
        self.assertEqual(result.sum(0), torch.ones(3, device=device))

    @onlyAccelerator
    @skipXPU  # Still here — known gap
    def test_matmul(self, device):
        a = torch.randn(3, 3, device=device)
        b = torch.randn(3, 3, device=device)
        self.assertEqual(a @ b, torch.matmul(a, b))

instantiate_device_type_tests(TestFooDevice, globals())
```

### Steps

1. **Scrutinize every CUDA reference.** Ask: "CUDA as device or CUDA as feature?" Most are the former → S2.
2. **Create `TestFooDevice` class** inheriting from `TestCase`.
3. **Add `device` parameter** as first arg after `self` on each test method.
4. **Replace hardcoded device strings**: `"cuda"` → `device` param, `.cuda()` → `.to(device)`.
5. **Enlarge whitelist, keep blacklist**: `@onlyCUDA` → `@onlyAccelerator`, `@unittest.skipIf(not TEST_CUDA, ...)` → `@onlyAccelerator`. Keep `@skipXPU`, `@skipCUDAIf`, `@skipMPS`, `@skipMeta`, `@onlyNativeDeviceTypesAnd` as-is.
6. **Replace device-specific APIs**: `torch.cuda.is_available()` → `torch.accelerator.is_available()`, Category A APIs → `torch.accelerator.*` equivalents (see catalog).
7. **Register**: `instantiate_device_type_tests(TestFooDevice, globals())` at module level.
8. **Remove stale imports**: `TEST_CUDA`, `TEST_MPS` only if no longer referenced.

### Key Rules

- **`@onlyAccelerator` is a method decorator, NOT a class decorator.** Applied to a class, it replaces the class with a function and `instantiate_device_type_tests` fails.
- **Use device-type-aware skips in S2 classes**: `skipXPUIf(True, msg)` / `skipCUDAIf(condition, msg)` from `common_device_type` (not `common_utils`) — these check `self.device_type` and only skip the specific device variant.
- **Category A APIs** (`empty_cache`, `synchronize`, `CUDAGraph`, `memory_*`) have `torch.accelerator.*` equivalents — they do NOT make a test CUDA-specific.
- **Category B APIs** (`Stream`, `Event`) are general concepts on all backends — they do NOT make a test CUDA-specific.

## Strategy 3: Accelerator-Specific (S3)

Tests requiring a particular accelerator's unique (Category C) features.

**Preferred Pattern — `instantiate_device_type_tests` with `only_for`:**
Use this when the class has `@dtypes`, `@dtypesIfCUDA`, `@dtypesIfCPU`, or `@parametrize` decorators — these rely on device-type injection from `instantiate_device_type_tests`.
```python
class TestFooCUDA(TestCase):
    # device is injected by instantiate_device_type_tests (always "cuda")
    # @dtypesIfCUDA resolves correctly because device_type is known

    @dtypesIfCUDA(torch.float16, torch.float32)
    def test_cuda_specific_feature(self, device, dtype):
        # device == "cuda" always
        torch.cuda.empty_cache()
        t = torch.randn(100, 100, device=device, dtype=dtype)
        ...

# Module level:
instantiate_device_type_tests(TestFooCUDA, globals(), only_for="cuda")
```
This pattern:
- Injects `device="cuda"` into every test method (no hardcoded `device = "cuda"`)
- Correctly resolves `@dtypesIfCUDA`/`@dtypesIfCPU`/`@dtypes` (device-type-aware decorators)
- Eliminates per-method `@onlyCUDA` decorators
- Keeps mechanism consistency with S2 (`instantiate_device_type_tests`)

**Fallback — Plain TestCase with setUp** (no device-type-aware decorators):
```python
class TestFooCUDA(TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_cuda_stream(self):
        s = torch.cuda.Stream()
        ...
```

**Why NOT `@instantiate_parametrized_tests`?**
`@instantiate_parametrized_tests` cannot resolve `@dtypesIfCUDA`/`@dtypesIfCPU` correctly — these decorators rely on the device-type context that only `instantiate_device_type_tests` provides. Using `@instantiate_parametrized_tests` for S3 classes with device-type-aware decorators results in incorrect dtype selection or runtime errors.

**Steps:**
1. Confirm the test genuinely uses Category C APIs.
2. Extract into `TestFoo<Device>` class with descriptive name.
3. Add `device` parameter to each test method (injected by `instantiate_device_type_tests`).
4. Use `instantiate_device_type_tests(TestFooCUDA, globals(), only_for="cuda")` at module level — preferred if any `@dtypes`/`@dtypesIfCUDA`/`@parametrize` decorators exist.
5. Fallback: plain `TestCase` with `setUp` guard and hardcoded `device = "cuda"` — only when NO device-type-aware decorators exist.

## Combined Workflow

### Step 1: Audit
Classify every test method. Create a table:

| Test Method | Device Usage | Category | Target Strategy |
|-------------|-------------|----------|-----------------|
| `test_basic_add` | None | unrelated | S1 |
| `test_softmax_cuda` | Generic only | agnostic | S2 |
| `test_cuda_stream` | CUDA-specific | specific | S3 |

### Step 2: Split
Create up to three classes following the naming convention and patterns above.

### Step 3: Clean up
- Remove stale `TEST_CUDA`/`TEST_MPS` imports and `copy_tests()` calls
- Remove `device` parameter from S1 tests
- **Keep blacklist skips** (`@skipXPU`, `@skipMPS`, `@skipMeta`, `@skipCUDAIf`, `@onlyNativeDeviceTypesAnd`)

### Step 4: Update external references after class renames

When a class is renamed (e.g., `TestCommon` → `TestCommonDevice`), external references to the old class name will **silently stop matching**. This causes previously-skipped tests to run and fail, or expected failures to become unguarded.

**Three locations to check:**

**(a) DecorateInfo in `common_methods_invocations.py`** — `DecorateInfo` entries use exact `cls_name` comparison:

```bash
python -c "
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import DecorateInfo
old = {'TestOldName1', 'TestOldName2'}
for op in op_db:
    for d in op.decorators:
        if isinstance(d, DecorateInfo) and d.cls_name in old:
            print(f'{op.name}: cls_name={d.cls_name}, test_name={d.test_name}')
"
```

**Fix:** Search-and-replace the old class name in `common_methods_invocations.py`. For class splits, verify which new class owns each test method first.

**(b) `test/dynamo_skips/`** — filenames are `ClassName.test_method_name`. When a class is renamed, old filenames no longer match and skipped tests may start running:

```bash
# Find stale entries after renaming TestFoo -> TestFooDevice
ls test/dynamo_skips/TestFoo.* 2>/dev/null
```

**Fix:** Rename files to use the new class name: `mv test/dynamo_skips/TestFoo.test_x test/dynamo_skips/TestFooDevice.test_x`

**(c) `test/dynamo_expected_failures/`** — same filename convention as dynamo_skips:

```bash
# Find stale entries after renaming TestFoo -> TestFooDevice
ls test/dynamo_expected_failures/TestFoo.* 2>/dev/null
```

**Fix:** Same as (b) — rename files to match the new class name.

### Step 5: Verify
1. **Test count**: `grep -c "def test_" test/test_file.py` — must match original
2. **Class structure**: `grep "^class " test/test_file.py` — verify naming and instantiation
3. **DecorateInfo**: Step 4(a) check script produces zero output
4. **dynamo_skips**: Step 4(b) check produces no stale entries
5. **dynamo_expected_failures**: Step 4(c) check produces no stale entries
6. **Syntax**: `python -c "import py_compile; py_compile.compile('test/test_file.py', doraise=True)"`

## Instantiation Mechanism Comparison

| Mechanism | Creates Device Variants? | Generic Class Discoverable? | Use When |
|-----------|--------------------------|----------------------------|----------|
| Plain `TestCase` | No | Yes | No parametrization needed |
| `instantiate_parametrized_tests()` | No | Yes | Tests with `@parametrize`/`@ops`/`@dtypes`, no device dependency |
| `instantiate_device_type_tests()` | Yes (CPU, CUDA, MPS, ...) | No (removed from scope) | Tests with a `device` parameter, works on any accelerator |
| `instantiate_device_type_tests(..., only_for="cuda")` | Yes (CUDA only) | No (removed from scope) | S3 classes with `@dtypes`/`@dtypesIfCUDA`/`@dtypesIfCPU`; device injected as `"cuda"` |

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| **Removing blacklist skips** (`@skipXPU`, `@skipCUDAIf`, `@skipMPS`, `@skipMeta`, `@onlyNativeDeviceTypesAnd`) | Keep as-is — they document known gaps |
| **Treating Cat A/B APIs as CUDA-specific** (`empty_cache`, `synchronize`, `CUDAGraph`, `Stream`, `Event`, `memory_*`) | These are S2 — consult `device_api_catalog.yaml` |
| **`@onlyAccelerator` as class decorator** | Use as **method decorator** only — on a class it replaces the class with a function |
| **Using `skipIfXpu`/`skipIfCUDA` from `common_utils` in S2 classes** | Use `common_device_type` equivalents (`skipXPUIf`, `skipCUDAIf`) — they check `self.device_type` and only skip the target variant |
| **Naming S1 class with device suffix** (e.g., `TestFooCPU`) | Use original name without suffix (`TestFoo`) — S1 has no device dependency |
| **Moving cross-device tests (CPU+GPU) to S1** | Tests using both CPU and GPU tensors still need a GPU — keep in S2 |
| **Renaming class without updating DecorateInfo** | Search `common_methods_invocations.py` for old class name and update |
| **Renaming class without updating dynamo_skips/** | Search `test/dynamo_skips/` for filenames starting with old class name and rename to new class name |
| **Renaming class without updating dynamo_expected_failures/** | Search `test/dynamo_expected_failures/` for filenames starting with old class name and rename to new class name |
| **Using `instantiate_device_type_tests` for S1 tests** | Creates wasteful per-device variants doing the same CPU work — use `instantiate_parametrized_tests` |
| **Using `@instantiate_parametrized_tests` for S3 with `@dtypesIfCUDA`** | `@dtypesIfCUDA` needs device-type context from `instantiate_device_type_tests` — use `instantiate_device_type_tests(TestFooCUDA, globals(), only_for="cuda")` |
| **Mixing `device` param and hardcoded `"cuda"` in same class** | Pick one strategy per class |

## Related Skills

- `agent/skills/classify-test-files` — Scan and classify test files before refactoring
