# Dispatch Coverage

How to determine whether XPU already has usable coverage for an operator.

XPU code lives in **two repositories** — both are in scope:
- `intel/torch-xpu-ops` — XPU kernels, SYCL implementations, fallback, backend YAML overrides
- `pytorch/pytorch` — upstream XPU backend integration, dispatch, codegen, shared native code, tests

**Upstream migration**: XPU ops are progressively moving upstream to `pytorch/pytorch`. When the same op has implementations in both repos, the upstream version takes precedence at runtime. A torch-xpu-ops implementation may be shadowed.

## Where To Look (intel/torch-xpu-ops)

1. `yaml/native/native_functions.yaml` — schema and dispatch key configuration
2. `src/ATen/native/xpu/XPUFallback.template` — fallback list
3. `src/ATen/native/xpu/` — host-side glue and registration
4. `src/ATen/native/xpu/sycl/` — SYCL kernel implementations
5. `yaml/xpu_functions.yaml` — auxiliary metadata (lowest priority)

## Where To Look (pytorch/pytorch)

1. `aten/src/ATen/native/native_functions.yaml` — authoritative schema and dispatch keys
2. `aten/src/ATen/native/xpu/` — upstream XPU native code
3. `tools/autograd/derivatives.yaml` — backward formulas
4. `torch/_decomp/` and `torch/_refs/` — decomposition registration
5. `aten/src/ATen/native/cuda/` — CUDA peer implementations
6. `aten/src/ATen/native/transformers/cuda/` and `…/xpu/`
7. `test/xpu/` — upstream XPU tests

These files are not in the local workspace for single-op analysis — access them via GitHub tools, a local pytorch checkout, or web fetch. In batch scan mode (full/daily), these files are available in the cloned pytorch repo. If upstream files are unreachable, note the gap and proceed with local evidence only.

## Coverage Signals

| Signal | Means | Priority when this is the only coverage |
|--------|-------|----------------------------------------|
| XPU dispatch key in backend YAML | Native XPU path | n/a (has native support) |
| Source-backed `TORCH_IMPL_FUNC` | Landed XPU implementation | n/a (has native support) |
| `structured_delegate: foo.out` | Judge by delegate target | n/a (has native support) |
| `CompositeImplicitAutograd` / `CompositeExplicitAutograd` | Generic runtime path | n/a (has native support) |
| Decomposition registered | Coverage via decomp | n/a (has native support) |
| In `XPUFallback.template` explicit `fallback_list` | Routes to CPU fallback | Report as "fallback only" (low) |
| In `XPUFallback.template` global backend fallback | CPU fallback only with `PYTORCH_ENABLE_XPU_FALLBACK=1` | Report as "fallback only" (low) |
| Only in `yaml/xpu_functions.yaml` | Metadata, not runtime | Not coverage |
| Only a test skip/xfail | Test annotation, not code | Not coverage — report as signal |

## Key Interpretations

**Fallback**: `XPUFallback.template` contains two mechanisms — (1) explicit per-op fallback registration that always routes to `xpu_fallback_impl` (CPU), and (2) a global backend fallback whose default path is `TORCH_CHECK_NOT_IMPLEMENTED` and only becomes CPU fallback when `PYTORCH_ENABLE_XPU_FALLBACK=1`. Both indicate XPU lacks a native GPU implementation for that op. Report as "fallback only" (low priority).

**How to detect silent CPU fallback at runtime**: run with `TORCH_SHOW_DISPATCH_TRACE=1` and check whether the op dispatches to XPU or falls back to CPU. Also check output tensor `.device` after the op.

**Structured delegate**: `structured_delegate: foo.out` means support is judged by `foo.out`, not the wrapper. Missing a hand-written XPU file for the wrapper is not evidence of missing support.

**Wrapper inheritance**: Thin frontend shims that redispatch to `*.out` or normalize arguments before a shared implementation do not need separate XPU registration.

**`HasDecompTest.test_has_decomposition.expect`**: Presence means decomposition is *missing*. This file is a negative snapshot.

**Dtype coverage**: XPU dtype support varies by hardware generation. FP64 may not be available on client GPUs (Intel Arc). Do not assume all CUDA dtypes map 1:1 to XPU.
