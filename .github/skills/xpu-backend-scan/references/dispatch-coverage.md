# Dispatch Coverage

How to determine whether XPU already has usable coverage for an operator.

XPU code lives in **two repositories** — both are in scope:
- `intel/torch-xpu-ops` — XPU kernels, SYCL implementations, fallback, backend YAML overrides
- `pytorch/pytorch` — upstream XPU backend integration, dispatch, codegen, shared native code, tests

## Priority Sources (intel/torch-xpu-ops)

Inspect in this order (this is lookup order, not coverage priority — see SKILL.md Step 3 for the decision rule):
1. `yaml/native/native_functions.yaml` — schema and dispatch key configuration
2. `src/ATen/native/xpu/XPUFallback.template` — fallback list
3. `src/ATen/native/xpu/` — host-side glue and registration
4. `src/ATen/native/xpu/sycl/` — SYCL kernel implementations
5. `yaml/xpu_functions.yaml` — auxiliary metadata (lowest priority)

## Priority Sources (pytorch/pytorch)

Also inspect — XPU backend code lives upstream too:
1. `aten/src/ATen/native/native_functions.yaml` — authoritative schema and dispatch keys
2. `aten/src/ATen/native/xpu/` — upstream XPU native code
3. `tools/autograd/derivatives.yaml` — backward formulas
4. `torch/_decomp/` and `torch/_refs/` — decomposition registration
5. `aten/src/ATen/native/cuda/` — CUDA peer implementations
6. `aten/src/ATen/native/transformers/cuda/` and `…/xpu/`
7. `test/xpu/` — upstream XPU tests

These files are not in the local workspace. Access them via GitHub tools, a local pytorch checkout, or web fetch. If upstream files are unreachable, note the gap and proceed with local evidence only.

## Coverage Signals

| Signal | Means | Blocks "missing impl"? |
|--------|-------|----------------------|
| In `XPUFallback.template` explicit `fallback_list` | Always routes to CPU fallback | Yes |
| In `XPUFallback.template` global backend fallback | Default path errors; CPU fallback becomes callable only with `PYTORCH_ENABLE_XPU_FALLBACK=1` | Only when enabled |
| XPU dispatch key in backend YAML | Native XPU path | Yes |
| `structured_delegate: foo.out` | Judge by delegate target | Yes |
| `CompositeImplicitAutograd` / `CompositeExplicitAutograd` | Generic runtime path | Yes |
| Decomposition registered | Coverage via decomp | Yes |
| Source-backed `TORCH_IMPL_FUNC` | Landed XPU implementation | Yes |
| Only in `yaml/xpu_functions.yaml` | Metadata, not runtime | No |
| Only a test skip/xfail | Test annotation, not code | No |

## Key Interpretations

**Fallback**: `XPUFallback.template` contains two mechanisms — (1) explicit per-op fallback registration that always routes to `xpu_fallback_impl` (CPU), and (2) a global backend fallback whose default path is `TORCH_CHECK_NOT_IMPLEMENTED` and only becomes CPU fallback when `PYTORCH_ENABLE_XPU_FALLBACK=1`. Treat the first as callable coverage. Treat the second as callable coverage only when the env-gated fallback is actually enabled. CPU fallback on a GPU op may support a defect finding (device-correctness risk).

**Structured delegate**: `structured_delegate: foo.out` means support is judged by `foo.out`, not the wrapper. Missing a hand-written XPU file for the wrapper is not evidence of missing support.

**Wrapper inheritance**: Thin frontend shims that redispatch to `*.out` or normalize arguments before a shared implementation do not need separate XPU registration.

**`HasDecompTest.test_has_decomposition.expect`**: Presence means decomposition is *missing*. This file is a negative snapshot.
