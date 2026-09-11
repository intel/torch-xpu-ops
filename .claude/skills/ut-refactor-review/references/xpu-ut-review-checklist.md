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

This checklist focuses on the XPU-specific and distributed concerns of a test
refactor. Generic test-refactoring checks (classification, naming, API
replacement, `HardwareClassification` tagging, instantiation mechanism) are out
of scope here.

## 1. Device gating precision (blanket-skip anti-pattern)

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `if device_type == "xpu": self.skipTest(...)` inside a test the PR enables on XPU | May be an over-skip: disables the whole XPU test to dodge one CUDA-specific check, hiding the real predicate. Human review required. | Info |
| `@unittest.skipIf(not SM70OrLater, ...)` | Decorators like `@unittest.skipIf(not SM70OrLater, ...)` used in non-CUDA specific test classes will erroneously skip XPU and other active devices. | Blocker |

## 2. Skips, xfails, and tolerances

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `@skipXPU` / `@xfailIf(TEST_XPU)` / `DecorateInfo(unittest.skip("Skipped"))` with no adjacent issue link | Every skip/xfail needs a tracking issue. A bare skip/xfail is untraceable. | Major |
| Issue link that does not match the failure, or omits an in-tree `pytorch/pytorch` issue | The linked issue must describe the same failure the skip/xfail works around; prefer a `pytorch/pytorch` issue when the failure is in-tree. | Minor |
| `skip` used where the test should pass but currently fails; `skip` used for a pure numeric mismatch | Wrong mechanism hides regressions. Unsupported capability -> `skipIf`/`@skipXPU`; should-pass-but-fails -> `xfailIf`/`expectedFailure`; numeric drift -> `DecorateInfo(toleranceOverride({dtype: tol(...)}), ..., device_type='xpu')`. | Major |
| `device_type='xpu'` skip with no `dtypes=(...)` for a single-dtype failure | Over-skips beyond the actual failure. Narrow the scope to the failing dtype/device. | Minor |
| Old `@skipIfXpu` / skip left in place though the PR says the op now passes | Stale skips must be removed; reviewers actively push to un-skip ops that now pass. | Minor |

## 3. Test intent and coverage preservation

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| A `*_cuda` variant merged into a generic test with fewer inputs/branches | Confirm the union of inputs/branches is preserved, not a subset. | Major |
| `assertEqual` weakened, a `gradcheck` removed, or expected values changed during generalization | Generalization must not weaken assertions or change expected results. | Major |
| Helper moved into a mixin/base class with an altered body | Helper extraction must be behavior-preserving; confirm no method body changed and both classes still reach it. | Major |
| Class-level decorator / `onlyNativeDeviceTypesAnd([...])` that omits the device the PR claims to enable | The "enabled" test then silently does not run on XPU. Verify the device is actually included. | Blocker |
| Tests moved out of a subclassed test class; `setUp` in the new class missing an attribute the tests read | Class splits can drop inherited parametrized instances (e.g. a `persistent_workers=True` subclass matrix) and `AttributeError` on missing state (a stale `@expectedFailure` then goes falsely green). A stable `grep -c 'def test_'` count does NOT prove coverage is unchanged. | Blocker |
| `x.is_cuda` widened to `x.is_cuda or x.is_xpu` | Loosening a device-residency assertion is permitted, but confirm it still asserts the tensor is on the *expected* device, not silently accepting any device. | Minor |
| `if TEST_CUDA: mem_stats["active_bytes.all.peak"]` widened to `TEST_CUDA or TEST_XPU` | Assumes XPU exposes the same stats key as CUDA. Confirm the key exists for XPU rather than assuming parity, or it `KeyError`s. | Blocker |
| Test exercises an op with only a CUDA/Meta registration, no XPU one | Some tests only run on XPU if a backing registration is added (C++ `TORCH_LIBRARY_IMPL(..., XPU, m)`, `torch.library.impl(..., "XPU")`, `register_autocast(..., "xpu", ...)`). Missing it makes the "enabled" test fail or silently no-op. | Blocker |

## 4. Cross-device blast radius

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `only_for=(...)` changed (e.g. dropping `"hpu"`) | A dropped backend gets zero execution. Confirm the new tuple is intentional and does not silently drop an existing backend. | Blocker |
| `DecorateInfo(...)` added/edited without `device_type=` | An unscoped decorator applies to all backends and can change CUDA/MPS/HPU/CPU behavior. Confirm `device_type=` is set. | Blocker |
| Reordered `skips=`/`decorators=` tuples or changed `active_if` | Can silently alter another backend (MPS/HPU); a real reviewer concern. | Blocker |
| An iterable feeding `@parametrize` converted `tuple -> set` (or `set -> tuple`) | Introduces nondeterministic ordering; has caused real breakage needing a follow-up fix. Flag any such conversion of a parametrization source. | Blocker |
| New module-level `instantiate_device_type_tests` call or `torch._lazy` init at import time | Module-level side effects can perturb other tests in the file. | Major |

## 5. Device-agnostic backend selection (multi-backend tests)

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `requires_nccl()` / `init_process_group(backend="nccl")` | Hardcoded backend requirements block XPU (zero execution on XPU). Use `requires_accelerator_dist_backend([...])` and resolve the backend from the device. | Blocker |
| Literal `"cpu:gloo,cuda:nccl"` backend string | Build it from the device (`get_default_backend_for_device(device_type)`). It returns a *single* backend, so a combined `"cpu:gloo,<device>:<backend>"` string must be assembled explicitly, not collapsed to one backend. | Major |

## 6. Refactor purity

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| A PR titled "refactor" that also adds `allow_xpu=True` or a new `instantiate_device_type_tests` device, or swaps `@requires_cuda` for `@onlyAccelerator` | A refactor must have no functional change; `@onlyAccelerator` is NOT equivalent to `@requires_cuda` (it changes which devices run). Move enablement to a follow-up PR. | Major |
| `@decorateIf(..., lambda params: params["device"] == torch.device("cpu"))` after moving to `instantiate_device_type_tests` | The harness passes `device` as a string, so the predicate must compare `== "cpu"` or it silently never matches. | Major |

## 7. Decorator parity (CUDA -> XPU mirroring)

When a CUDA test is ported to XPU, device-conditional decorators keyed to
`device_type="cuda"` only affect the CUDA instantiation, so XPU needs its own
decorator keyed to `"xpu"` **with the same scope** (same dtypes/size/condition),
or a deliberate, documented decision not to mirror it. Parity is not "copy every
CUDA decorator"; it is "for each CUDA-conditional behavior, decide whether XPU
needs it and, if so, express it with the same scope."

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `@largeTensorTest("20GB", "cuda")` with no `@largeTensorTest("20GB", "xpu")` | The memory guard checks free memory on the named device. Without the XPU copy the port checks the wrong device or runs an OOM-prone test unguarded. Mirror it with the exact same size string. | Blocker |
| `@onlyCUDA` left on a test meant to also cover XPU | The test never instantiates for XPU. Decide intent: add a parallel `@onlyXPU` test or broaden to run on both. | Blocker |
| `@dtypesIfCUDA(...)` (or old `@dtypeIfCuda`) with no `@dtypesIfXPU(...)` | XPU silently falls back to the base `@dtypes` set. Mirror with the same dtype set; narrow it only for dtypes XPU genuinely lacks, and note why. | Major |
| `@skipCUDAIf(...)` / `@skipCUDAIfNoMagma` / `NoCusolver` / `NoCudnn` / ROCm / MIOpen mirrored blindly to XPU | Feature-gated CUDA-stack skips (Magma, cuSOLVER, cuDNN, ROCm) usually have no XPU meaning. Mirror to `@skipXPUIf` only if the same underlying limitation applies to XPU; otherwise drop it or replace it with the matching XPU capability check. | Major |
| `@expectedFailureXPU` added without confirming the failure reproduces on XPU | A spurious expected-failure hides a real regression (the test would silently pass-as-xfail). Verify the failure actually occurs on XPU before mirroring an `expectedFailure`. | Major |
| `@precisionOverride`/`@toleranceOverride` inherited unchanged on the XPU run | These are not device-keyed. Confirm XPU is not stuck with a CPU-strict tolerance its hardware can't meet, nor an over-loose CUDA-tuned one; re-tune per device as needed. | Minor |
| Multiple CUDA-conditional decorators stacked on one method (`@dtypesIfCUDA` + `@precisionOverride` + `@skipCUDAIf`) | Each decorator is an independent parity obligation. Check every decorator in the stack; a common miss is mirroring the dtype override but forgetting the skip. | Minor |
| `@tf32_on_and_off` / `@with_tf32_off` mechanically mirrored to XPU | TF32 is a CUDA (Ampere+) concept with no direct XPU equivalent. Do not mirror; instead confirm the XPU port runs in a well-defined numeric mode and has appropriate tolerance handling. | Info |
