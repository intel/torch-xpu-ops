# Dispatch Coverage

How to determine whether XPU already has usable coverage for an operator.

XPU code lives in **two repositories** — both are in scope:
- `intel/torch-xpu-ops` — XPU kernels, SYCL implementations, fallback, backend YAML overrides
- `pytorch/pytorch` — upstream XPU backend integration, dispatch, codegen, shared native code, tests

## Priority Sources (intel/torch-xpu-ops)

Inspect in this order (this is lookup order, not coverage priority — see SKILL.md Step 3 for override rules):
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

## Coverage Signals

| Signal | Means | Blocks "missing impl"? |
|--------|-------|----------------------|
| In `XPUFallback.template` | Runtime callable via CPU | Yes |
| XPU dispatch key in backend YAML | Native XPU path | Yes |
| `structured_delegate: foo.out` | Judge by delegate target | Yes |
| `CompositeImplicitAutograd` / `CompositeExplicitAutograd` | Generic runtime path | Yes |
| Decomposition registered | Coverage via decomp | Yes |
| Source-backed `TORCH_IMPL_FUNC` | Landed XPU implementation | Yes |
| Only in `yaml/xpu_functions.yaml` | Metadata, not runtime | No |
| Only a test skip/xfail | Test annotation, not code | No |

## Key Interpretations

**Fallback**: blocks a "missing implementation" conclusion but may support a defect finding (CPU fallback on a GPU op is a device-correctness risk).

**Structured delegate**: `structured_delegate: foo.out` means support is judged by `foo.out`, not the wrapper. Missing a hand-written XPU file for the wrapper is not evidence of missing support.

**Wrapper inheritance**: Thin frontend shims that redispatch to `*.out` or normalize arguments before a shared implementation do not need separate XPU registration.

**`HasDecompTest.test_has_decomposition.expect`**: Presence means decomposition is *missing*. This file is a negative snapshot.
