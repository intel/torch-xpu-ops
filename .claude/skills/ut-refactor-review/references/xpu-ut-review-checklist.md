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
| `@skipXPU` / `@skipIfXpu` / `@xfailIf(TEST_XPU)` / `DecorateInfo(unittest.skip("Skipped"))` without tracking issue link | Every skip/xfail needs a tracking issue. A bare skip/xfail is untraceable. | Major |
| `device_type='xpu'` skip with no `dtypes=(...)` for a single-dtype failure | May be an over-skip beyond the actual failure. Narrow the scope to the failing dtype/device. | Info |
| Old `@skipIfXpu` / skip left in place when the test already passes on XPU | Stale skips must be removed. | Minor |

## 3. Test intent and coverage preservation

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| Helper moved into a mixin/base class with an altered body | Helper extraction must be behavior-preserving; confirm no method body changed and both classes still reach it. | Major |
| `if TEST_CUDA: < code block >` widened to `TEST_CUDA or TEST_XPU` | Assumes XPU exposes the same stats key as CUDA. Confirm the key exists for XPU rather than assuming parity, or it `KeyError`s. | Blocker |
| Test exercises an op with only a CUDA/Meta registration, no XPU one | An op is mistakenly identified as being supported on XPU, which makes the "enabled" test fail or silently no-op. | Blocker |

## 4. Cross-device blast radius

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| Reordered `skips=`/`decorators=` tuples or changed `active_if` | Can silently alter another backend (MPS/HPU); a real reviewer concern. | Blocker |
| An iterable feeding `@parametrize` converted `tuple -> set` (or `set -> tuple`) | Introduces nondeterministic ordering; has caused real breakage needing a follow-up fix. Flag any such conversion of a parametrization source. | Blocker |

## 5. Device-agnostic backend selection (multi-backend tests)

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `requires_nccl()` / `init_process_group(backend="nccl")` | Hardcoded backend requirements block XPU (zero execution on XPU). Use `requires_accelerator_dist_backend([...])` and resolve the backend from the device. | Blocker |
| Literal `"cpu:gloo,cuda:nccl"` backend string | Build it from the device (`get_default_backend_for_device(device_type)`). It returns a *single* backend, so a combined `"cpu:gloo,<device>:<backend>"` string must be assembled explicitly, not collapsed to one backend. | Major |

## 6. Decorator parity (CUDA -> XPU mirroring)

Mirror CUDA decorators to XPU only when needed, keeping the same scope; don’t copy them mechanically.

| Code Pattern | What It Means | Severity |
| --- | --- | --- |
| `@onlyCUDA` (or a CUDA-only test class) left on a case whose computation is device-generic and could run on other accelerators | A reusable CUDA case should not stay CUDA-only; it must be enabled for XPU. Broaden it to run on XPU (e.g. `@onlyOn(["cuda", "xpu"])` or a device-agnostic class instantiated with `allow_xpu=True`), or add a parallel `@onlyXPU` case. Keep `@onlyCUDA` only when the case genuinely exercises a CUDA-specific feature. | Blocker |
| `@largeTensorTest("20GB", "cuda")` (or `device="cuda"`, or bare on a device-generic test whose primary device is CUDA) with no XPU counterpart | The memory guard checks free memory on the named device. XPU also needs a memory guard, using the same size as CUDA or a human-decided different size if needed. | Blocker |
| `@dtypesIfCUDA(...)` (or old `@dtypeIfCuda`) with no `@dtypesIfXPU(...)` | XPU silently falls back to the base `@dtypes` set. Mirror with the same dtype set; narrow it only for dtypes XPU genuinely lacks, and note why. | Major |
| `@skipCUDAIf(...)` / `@skipCUDAIfNoMagma` / `NoCusolver` / `NoCudnn` / ROCm / MIOpen mirrored blindly to XPU | Feature-gated CUDA-stack skips (Magma, cuSOLVER, cuDNN, ROCm) usually have no XPU meaning. Mirror to `@skipXPUIf` only if the same underlying limitation applies to XPU; otherwise drop mirroring or use the matching XPU capability check. | Major |
| `@expectedFailureXPU` added without confirming the failure reproduces on XPU | Please verify the failure actually occurs on XPU before mirroring an `expectedFailure`. | Info |
| `@precisionOverride`/`@toleranceOverride` inherited unchanged on the XPU run | Tolerance should be re-calibrated based on actual XPU behavior to avoid being too strict or too loose, rather than copied directly. | Minor |
| Multiple CUDA-conditional decorators stacked on one method (`@dtypesIfCUDA` + `@precisionOverride` + `@skipCUDAIf`) | Each decorator is an independent parity obligation. Check every decorator in the stack, not just the first; a common miss is mirroring the dtype override but forgetting the skip. | Info |
| `@tf32_on_and_off` / `@with_tf32_off` reused on an XPU test | These decorators are CUDA-keyed (`torch.cuda.is_tf32_supported()` and `device.type == "cuda"`). Currently, XPU controls TF32 with use `torch.backends.mkldnn.flags(allow_tf32=...)` / `torch.backends.mkldnn.fp32_precision` (see the local `tf32_on_and_off` in `test/xpu/test_gemm.py`). | Major |
| Decorators mirrored to XPU but `instantiate_device_type_tests(...)` does not include XPU (no `allow_xpu=True`, or `only_for`/`except_for` excludes `"xpu"`) | The mirrored decorators only take effect if the class is actually instantiated for XPU. If XPU is not in the instantiated device list, none of the parity below runs no matter how carefully it was mirrored. | Blocker |
