# Dispatch Coverage

How to determine whether XPU already has usable coverage for an operator.

## Priority Sources (torch-xpu-ops)

Inspect in this order (this is lookup order, not coverage priority — see SKILL.md Step 3 for priority):
1. `src/ATen/native/xpu/XPUFallback.template` — fallback list
2. `yaml/native/native_functions.yaml` — backend dispatch keys
3. `src/ATen/native/xpu/` — host-side glue
4. `src/ATen/native/xpu/sycl/` — SYCL kernel implementations
5. `yaml/xpu_functions.yaml` — auxiliary metadata (lowest priority)

## Priority Sources (upstream pytorch/pytorch)

Consult when peer semantics or schema confirmation is needed:
1. `aten/src/ATen/native/native_functions.yaml` — authoritative schema
2. `tools/autograd/derivatives.yaml` — backward formulas
3. `torch/_decomp/` and `torch/_refs/` — decomposition registration
4. `aten/src/ATen/native/cuda/` — CUDA implementations
5. `aten/src/ATen/native/transformers/cuda/` and `…/xpu/`

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
