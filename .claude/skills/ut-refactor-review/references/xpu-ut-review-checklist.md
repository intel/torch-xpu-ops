# XPU UT Refactor Review Checklist

Evaluate every changed line against these patterns. Each row gives a code
pattern to look for, what it means (the defect and the correct form), and a
severity. Severities follow this scale:

- **Blocker** — XPU gets wrong or zero test execution (the test never
  instantiates, runs on the wrong device, silently runs the wrong dtype/skip
  gating, risks an unguarded OOM/crash), or the change breaks another backend.
  Must fix before merge.
- **Major** — coverage or scope is silently wrong (a dropped/weakened assertion
  or case, a mismatched or missing skip/xfail, a behavior-changing "refactor").
  Fix before merge.
- **Minor** — tuning- or convention-level issue (tolerance, over-broad skip
  scope, naming/classification mismatch, stale skip). Fix or justify.
- **Info** — no direct action required; guidance, or a pattern with no XPU
  equivalent. Note and move on.

Rows are grouped by area and roughly ordered by how often they surfaced as real
review comments across the studied PRs.

## 1. Device generalization (replacing CUDA hardcoding)

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `x.cuda()` / `.cuda(rank)` inside a generalized test | Runs on CUDA (or errors on an XPU-only box) regardless of the parameterized device. Use `.to(device)` / `.to(device_type)` / `.to(rank)`. | Blocker |
| Literal `"cuda"` in `device=`, `torch.device("cuda")`, `init_device_mesh("cuda", ...)` | A hardcoded device where a `device`/`device_type` parameter is available defeats generalization. | Blocker |
| `@unittest.skipIf(not TEST_CUDA, ...)` on a test the PR enables on XPU | A CUDA-only gate keeps the test from running on XPU. Use `@onlyAccelerator` / `@onlyNativeDeviceTypesAnd([...])` when genuinely accelerator-generic. | Blocker |
| `torch.cuda.set_device` / `current_device` / `synchronize` / `current_stream` / `Event` / `Stream()` + `stream(s)` / `manual_seed`; `GradScaler(device="cuda")` | Prefer the full generic API: `torch.accelerator.set_device_index` / `current_device_index` / `synchronize` / `current_stream`, `torch.Event`, `torch.Stream()` + `with s:`, `device_module.manual_seed`, `GradScaler(device=device_type)`. Reviewers consistently push this, not just `is_available`/`device_count`/`current_accelerator`. | Minor |
| `if TEST_CUDA: mem_stats["active_bytes.all.peak"]` widened to `TEST_CUDA or TEST_XPU` | Assumes XPU exposes the same stats key as CUDA. Confirm the key exists for XPU rather than assuming parity, or it `KeyError`s. | Blocker |

## 2. Device gating precision (blanket-skip anti-pattern)

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `if device_type == "xpu": self.skipTest(...)` inside a test the PR enables on XPU | Skips the whole test for XPU just to route around one CUDA-specific check; over-skips and hides the real predicate. Gate only that check on its true condition and name the device (`device_type == "cuda" and not sm_is_or_higher_than(...)`), so XPU keeps the rest of the coverage. A genuinely CUDA-only test not being enabled on XPU should keep its CUDA scoping untouched, not gain an XPU skip. | Blocker |
| `@unittest.skipIf(not SM70OrLater, ...)` (device-agnostic arch gate) | A CUDA arch check left device-agnostic wrongly skips XPU and every non-CUDA device. Scope it: `@skipCUDAIf(not SM70OrLater, ...)`. Gate the XPU side on its own capability constant (`PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU`, an `Xe*OrLater`-style flag), not the CUDA `SM*` predicate. | Blocker |

## 3. `instantiate_device_type_tests` wiring

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `instantiate_device_type_tests(Cls, globals())` with `"xpu"` in `only_for` but no `allow_xpu=True` | XPU is not actually enabled; `allow_xpu=True` is required. Without it none of the other rules take effect. | Blocker |
| `only_for=(...)` changed (e.g. dropping `"hpu"`) | A dropped backend gets zero execution. Confirm the new tuple is intentional and does not silently drop an existing backend. | Blocker |
| Method still reads a module-level device or env var (`get_test_device()`, `LTC_TS_CUDA`) after moving under the harness | The class must be device-parameterized: methods take `self, device` (and `dtype`/`op` where relevant), or the test runs on the wrong device. | Blocker |
| `hw_classification` mismatched to the class scope | Device-agnostic (CPU + device) -> `GENERIC` (NOT instantiated via `instantiate_device_type_tests`); accelerator-instantiated -> `ACCELERATOR`; single-backend -> `CPU`/`CUDA`. A wrong tag can leave a class un-instantiated or double-instantiated, silently changing which devices run. | Major |

## 4. Skips, xfails, and tolerances

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `@skipXPU` / `@xfailIf(TEST_XPU)` / `DecorateInfo(unittest.skip("Skipped"))` with no adjacent issue link | Every skip/xfail needs a tracking issue. A bare skip/xfail is untraceable. | Major |
| Issue link that does not match the failure, or omits an in-tree `pytorch/pytorch` issue | The linked issue must describe the same failure the skip/xfail works around; prefer a `pytorch/pytorch` issue when the failure is in-tree. | Minor |
| `skip` used where the test should pass but currently fails; `skip` used for a pure numeric mismatch | Wrong mechanism hides regressions. Unsupported capability -> `skipIf`/`@skipXPU`; should-pass-but-fails -> `xfailIf`/`expectedFailure`; numeric drift -> `DecorateInfo(toleranceOverride({dtype: tol(...)}), ..., device_type='xpu')`. | Major |
| `device_type='xpu'` skip with no `dtypes=(...)` for a single-dtype failure | Over-skips beyond the actual failure. Narrow the scope to the failing dtype/device. | Minor |
| Old `@skipIfXpu` / skip left in place though the PR says the op now passes | Stale skips must be removed; reviewers actively push to un-skip ops that now pass. | Minor |

## 5. Test intent and coverage preservation

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| A `*_cuda` variant merged into a generic test with fewer inputs/branches | Confirm the union of inputs/branches is preserved, not a subset. | Major |
| `assertEqual` weakened, a `gradcheck` removed, or expected values changed during generalization | Generalization must not weaken assertions or change expected results. | Major |
| Helper moved into a mixin/base class with an altered body | Helper extraction must be behavior-preserving; confirm no method body changed and both classes still reach it. | Major |
| Class-level decorator / `onlyNativeDeviceTypesAnd([...])` that omits the device the PR claims to enable | The "enabled" test then silently does not run on XPU. Verify the device is actually included. | Blocker |
| Tests moved out of a subclassed test class; `setUp` in the new class missing an attribute the tests read | Class splits can drop inherited parametrized instances (e.g. a `persistent_workers=True` subclass matrix) and `AttributeError` on missing state (a stale `@expectedFailure` then goes falsely green). A stable `grep -c 'def test_'` count does NOT prove coverage is unchanged. | Blocker |
| `x.is_cuda` widened to `x.is_cuda or x.is_xpu` | Loosening a device-residency assertion is permitted, but confirm it still asserts the tensor is on the *expected* device, not silently accepting any device. | Minor |
| Test exercises an op with only a CUDA/Meta registration, no XPU one | Some tests only run on XPU if a backing registration is added (C++ `TORCH_LIBRARY_IMPL(..., XPU, m)`, `torch.library.impl(..., "XPU")`, `register_autocast(..., "xpu", ...)`). Missing it makes the "enabled" test fail or silently no-op. | Blocker |

## 6. Cross-device blast radius

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `DecorateInfo(...)` added/edited without `device_type=` | An unscoped decorator applies to all backends and can change CUDA/MPS/HPU/CPU behavior. Confirm `device_type=` is set. | Blocker |
| Reordered `skips=`/`decorators=` tuples or changed `active_if` | Can silently alter another backend (MPS/HPU); a real reviewer concern. | Blocker |
| An iterable feeding `@parametrize` converted `tuple -> set` (or `set -> tuple`) | Introduces nondeterministic ordering; has caused real breakage needing a follow-up fix. Flag any such conversion of a parametrization source. | Blocker |
| New module-level `instantiate_device_type_tests` call or `torch._lazy` init at import time | Module-level side effects can perturb other tests in the file. | Major |

## 7. Device-agnostic backend selection (multi-backend tests)

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `requires_nccl()` / `init_process_group(backend="nccl")` | Hardcoded backend requirements block XPU (zero execution on XPU). Use `requires_accelerator_dist_backend([...])` and resolve the backend from the device. | Blocker |
| Literal `"cpu:gloo,cuda:nccl"` backend string | Build it from the device (`get_default_backend_for_device(device_type)`). It returns a *single* backend, so a combined `"cpu:gloo,<device>:<backend>"` string must be assembled explicitly, not collapsed to one backend. | Major |

## 8. Refactor purity and class-split naming

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| A PR titled "refactor" that also adds `allow_xpu=True` or a new `instantiate_device_type_tests` device, or swaps `@requires_cuda` for `@onlyAccelerator` | A refactor must have no functional change; `@onlyAccelerator` is NOT equivalent to `@requires_cuda` (it changes which devices run). Move enablement to a follow-up PR. | Major |
| Split-class name/classification/instantiation that disagree | CPU-only -> `TestXxxCPU` (`CPU`), device-generic -> `TestXxxDevice` (`ACCELERATOR`) / `TestXxxGeneric` (`GENERIC`), CUDA-only -> `TestXxxCUDA` (`CUDA`). Name, classification, and instantiation must agree; reviewers frequently request these renames. | Minor |
| `@decorateIf(..., lambda params: params["device"] == torch.device("cpu"))` after moving to `instantiate_device_type_tests` | The harness passes `device` as a string, so the predicate must compare `== "cpu"` or it silently never matches. | Major |
