# XPU Upstream UT Review Checklist

Evaluate every changed line against these items. Each item names the defect to
look for and the correct pattern. Items are ordered by how often they surfaced
as real review comments in landed PRs.

## 1. Device generalization (replacing CUDA hardcoding)

- [ ] **No leftover `.cuda()`** — must be `.to(device)` / `.to(device_type)`.
  A `.cuda()` inside a generalized test silently runs on CUDA (or errors on an
  XPU-only box) regardless of the parameterized device.
- [ ] **No leftover literal `"cuda"`** in device strings, `init_device_mesh`,
  or `device=` kwargs where a device parameter is available.
- [ ] **`TEST_CUDA` -> generic** — `@unittest.skipIf(not TEST_CUDA, ...)` should
  become `@onlyAccelerator` (or `@onlyNativeDeviceTypesAnd([...])`) when the test
  is genuinely accelerator-generic. Do not keep a CUDA-only gate on a test the PR
  claims to enable on XPU.
- [ ] **Use the full generic accelerator API, not just availability queries** —
  reviewers consistently push `torch.accelerator.*` / `torch.get_device_module`
  everywhere a generic form exists, not only for `is_available` / `device_count`
  / `current_accelerator`. Flag leftover `torch.cuda.*`: `set_device` ->
  `set_device_index`, `current_device` -> `current_device_index`, `synchronize`,
  `current_stream`, `torch.cuda.Event` -> `torch.Event`, `torch.cuda.Stream()` +
  `stream(s)` -> `torch.Stream()` + `with s:`, `GradScaler(device="cuda")` ->
  `device=device_type`, `torch.cuda.manual_seed` -> `device_module.manual_seed`.
- [ ] **Memory / stats keys are validated per backend** — e.g. a branch that did
  `if TEST_CUDA: mem_stats["active_bytes.all.peak"]` becomes
  `if TEST_CUDA or TEST_XPU: ...` only if XPU actually exposes that key. Confirm
  the key exists for XPU rather than assuming CUDA parity.

## 2. Device gating precision (blanket-skip anti-pattern)

- [ ] **No blanket per-device skip inside a test the PR enables on XPU.** When a
  test is already generalized and run on XPU, do not skip the whole test for XPU
  just to route around one CUDA-specific check inside it. Gate only that check on
  its true condition, naming the device that lacks the capability, so XPU still
  gets the coverage the rest of the test provides. A whole-test XPU skip both
  over-skips and hides the real capability predicate. (A test that is genuinely
  CUDA-only and not being enabled on XPU should keep its existing CUDA scoping
  untouched, not gain an XPU skip.)
- [ ] **Arch/capability gates are device-scoped.** A CUDA arch check left as a
  device-agnostic `@unittest.skipIf(not SM70OrLater, ...)` wrongly skips XPU (and
  every non-CUDA device). Scope it: `@skipCUDAIf(not SM70OrLater, ...)`. When an
  XPU equivalent capability exists, gate the XPU side on its own constant
  (`PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU`, an `Xe*OrLater`-style flag) rather
  than reusing the CUDA `SM*` predicate.

## 3. `instantiate_device_type_tests` wiring

- [ ] **`allow_xpu=True` is present** on the `instantiate_device_type_tests`
  call for any class the PR claims to enable on XPU. Adding `"xpu"` to
  `only_for` without `allow_xpu=True` does not enable it.
- [ ] **`only_for` is consistent** — if the class previously ran on
  `("cuda", "hpu")`, confirm the new tuple is intentional and does not drop an
  existing backend.
- [ ] **The test class is device-parameterized** — methods take `self, device`
  (and `dtype`/`op` where relevant). A method still reading a module-level
  device or an env var (`get_test_device()`, `LTC_TS_CUDA`) after being moved
  under the harness is a bug.
- [ ] **`HardwareClassification` is correct** — device-agnostic (CPU + device)
  class -> `GENERIC`; class run through `instantiate_device_type_tests` on
  accelerators -> `ACCELERATOR`; a class that is genuinely single-backend ->
  `CPU` or `CUDA`. A `GENERIC` tag on a class full of accelerator-only tests (or
  vice versa) is wrong. A `GENERIC` class must NOT be instantiated via
  `instantiate_device_type_tests`.

## 4. Skips, xfails, and tolerances

- [ ] **Every skip/xfail has an adjacent tracking issue.** Bare
  `@skipXPU` / `@xfailIf(TEST_XPU)` / `DecorateInfo(unittest.skip("Skipped"))`
  with no issue link is a **Must Fix**.
- [ ] **Issue points at the right root cause.** Prefer a `pytorch/pytorch`
  issue when the failure is in-tree. The linked issue must describe the same
  failure the skip/xfail works around.
- [ ] **Correct mechanism:**
  - Capability genuinely unsupported (sleep kernel, a missing feature) ->
    `unittest.skipIf` / `@skipXPU`.
  - Test *should* pass but currently fails -> `xfailIf(TEST_XPU)` /
    `DecorateInfo(unittest.expectedFailure, ...)`. Using `skip` here hides the
    fix when it lands.
  - Numeric mismatch only -> `DecorateInfo(toleranceOverride({dtype: tol(...)}), 'TestClass', 'test_name', device_type='xpu')`. A skip for a tolerance issue
    is a **Must Fix**.
- [ ] **Scope is minimal.** `device_type='xpu'` and `dtypes=(...)` should be as
  narrow as the actual failure. A dtype-agnostic skip for a single-dtype failure
  over-skips.
- [ ] **Stale skips removed.** If the PR claims an op now passes, the
  corresponding old skip/`skipIfXpu` must be deleted. Reviewers actively push to
  un-skip ops that now pass (e.g. `max_unpool2d`).

## 5. Test intent and coverage preservation

- [ ] **No dropped test cases.** When a `*_cuda` variant is merged into a
  generic test, confirm the union of inputs/branches is preserved, not a subset.
- [ ] **Assertions unchanged.** Generalization must not weaken `assertEqual`,
  remove a `gradcheck`, or change expected values.
- [ ] **Helper extraction is behavior-preserving.** When helpers move into a
  mixin/base class (e.g. `_AutogradFunctionalHelpers`), confirm no method body
  changed and both classes still reach them.
- [ ] **The XPU path actually executes.** An inherited class-level decorator or
  an `onlyNativeDeviceTypesAnd` that omits the device can make the "enabled"
  test silently not run on XPU. Verify the device is included.
- [ ] **Class splits preserve coverage and required state.** When tests move out
  of a subclassed test class, they can lose inherited parametrized instances
  (e.g. a `persistent_workers=True` subclass matrix); a stable
  `grep -c 'def test_'` count does NOT prove coverage is unchanged. Also confirm
  the new class's `setUp` still sets every attribute the moved tests read (a
  missing `self.persistent_workers` will `AttributeError`, and a stale
  `@expectedFailure` can then go falsely green).
- [ ] **Device-residency assertion loosening is intentional and correct.**
  Widening `x.is_cuda` to `x.is_cuda or x.is_xpu` is a permitted assertion
  change, but confirm it still asserts the tensor is on the *expected* device
  rather than silently accepting any device.
- [ ] **Enablement prerequisites in non-test files exist.** Some tests only run
  on XPU if a backing registration is added: a C++
  `TORCH_LIBRARY_IMPL(..., XPU, m)` block, `torch.library.impl(..., "XPU")`, or
  `torch.library.register_autocast(..., "xpu", ...)`. If the test exercises such
  an op, confirm the XPU registration is present, otherwise the "enabled" test
  fails or silently no-ops.

## 6. Cross-device blast radius

- [ ] **OpInfo `DecorateInfo` edits are device-scoped.** An added/edited
  `DecorateInfo` must not change CUDA/MPS/HPU/CPU behavior unless intended.
  Confirm `device_type=` is set; an unscoped decorator applies to all backends.
- [ ] **MPS/HPU not collaterally affected.** Reordering `skips=`/`decorators=`
  tuples or changing an `active_if` can silently alter another backend. This was
  a real reviewer concern.
- [ ] **tuple -> set / set -> tuple changes.** Converting an iterable used by
  `@parametrize` to a `set` introduces nondeterministic ordering and has caused
  distributed-test breakage that needed a follow-up fix. Flag any such
  conversion of a parametrization source.
- [ ] **Module-level side effects.** New global `instantiate_device_type_tests`
  calls or `torch._lazy` init at import time must not perturb other tests in the
  file.

## 7. Device-agnostic backend selection (multi-backend tests)

- [ ] **Hardcoded backend requirements are generalized.** `requires_nccl()` ->
  `requires_accelerator_dist_backend([...])`; a hardcoded
  `init_process_group(backend="nccl")` -> a backend resolved from the device.
- [ ] **Backend strings are built from the device, not hardcoded.** A literal
  `"cpu:gloo,cuda:nccl"` should become device-driven (e.g. built from
  `torch.distributed.get_default_backend_for_device(device_type)`). Watch the
  semantics: `get_default_backend_for_device` returns a *single* backend, so a
  combined `"cpu:gloo,<device>:<backend>"` string must be assembled explicitly
  and not silently collapsed to one backend.

## 8. Refactor purity and class-split naming

- [ ] **A "refactor" PR must have no functional change.** If the title/description
  says refactor, it must not simultaneously add `allow_xpu=True` or a new
  `instantiate_device_type_tests` device. `@onlyAccelerator` is NOT equivalent to
  `@requires_cuda`; swapping them changes which devices run. Enablement belongs
  in a separate follow-up PR.
- [ ] **Split-class names match their scope and classification.** A CPU-only half
  should be `TestXxxCPU` (`HardwareClassification.CPU`), a device-generic half
  `TestXxxDevice` (`ACCELERATOR`) or `TestXxxGeneric` (`GENERIC`), a CUDA-only
  half `TestXxxCUDA` (`CUDA`). Reviewers frequently request these renames; the
  name, the classification, and the actual instantiation must agree.
- [ ] **`@decorateIf` predicates use the string device.**
  `instantiate_device_type_tests` passes `device` as a string, so predicates like
  `lambda params: params["device"] == torch.device("cpu")` must become
  `== "cpu"`, or they silently never match after the migration.

## Quick reference: correct patterns

```python
# Device-generic enablement
class TestFooDevice(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @onlyAccelerator
    def test_bar(self, device):
        x = torch.randn(4, device=device)          # not .cuda()

instantiate_device_type_tests(TestFooDevice, globals(), allow_xpu=True)

# Skip / xfail / tolerance are distinct mechanisms; each needs an issue link:
@unittest.skipIf(TEST_XPU, "...")                  # capability unsupported
@xfailIf(TEST_XPU)                                 # should pass, currently fails
DecorateInfo(toleranceOverride({torch.float64: tol(atol=9e-6, rtol=8e-7)}),
             "TestCommon", "test_numpy_ref", device_type="xpu")  # numeric drift
```
