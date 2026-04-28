# Dispatch And Layout

This reference condenses the dispatch, source-layout, and fallback guidance for
`xpu-backend-scan`.

## Coverage Precedence

When deciding whether XPU already has usable coverage, apply signals in this
order:

1. `src/ATen/native/xpu/XPUFallback.template`
2. `yaml/native/native_functions.yaml` entries with explicit XPU-family
   dispatch keys
3. structured delegate or structured/codegen paths that resolve to a verified
   XPU target
4. source-backed XPU registration, `TORCH_IMPL_FUNC`, or landed implementation
5. composite, decomposition, or shared helper coverage
6. `yaml/xpu_functions.yaml` metadata only

Lower-priority signals must not override a higher-priority positive coverage
signal.

## Where To Look

In the target `torch-xpu-ops` repository, inspect in roughly this order:

- `yaml/native/native_functions.yaml`
- `src/ATen/native/xpu/XPUFallback.template`
- `src/ATen/native/xpu/`
- `src/ATen/native/xpu/sycl/`
- `yaml/xpu_functions.yaml`
- `test/xpu/`

In upstream `pytorch/pytorch`, inspect when needed:

- `aten/src/ATen/native/native_functions.yaml`
- `tools/autograd/derivatives.yaml`
- `torch/_refs/__init__.py`
- `torch/_decomp/`
- `aten/src/ATen/native/cuda/`
- `aten/src/ATen/native/transformers/cuda/`
- `aten/src/ATen/native/transformers/xpu/`

## Important Interpretations

### Fallback

If an op appears in `XPUFallback.template`, runtime coverage exists.

- This blocks a Goal 3 missing-implementation conclusion.
- It can still support a Goal 1 finding because CPU fallback on XPU is a device
  correctness and performance risk.

### Structured And Delegate Paths

- `structured: true` means codegen may own the wrapper and registration shape.
- `structured_delegate: foo.out` means support should be judged by `foo.out`,
  not by the wrapper alone.
- Missing a hand-written XPU file is not enough to conclude missing support.

### Composite And Decomposition Paths

- `CompositeImplicitAutograd` and `CompositeExplicitAutograd` are runtime
  coverage, not missing-XPU evidence.
- decomposition registration is positive coverage evidence, but verify the real
  registration rather than relying on snapshot files alone.

### Auxiliary Metadata

`yaml/xpu_functions.yaml` is supporting metadata only.
It should not outweigh backend YAML, source-backed registration, structured
coverage, or fallback.

## Wrapper Inheritance

Some public overloads are thin frontend shims and should inherit support from a
different target instead of demanding a separate XPU-local registration.

Common cases:
- wrapper variants that redispatch to `*.out`
- frontend utility overloads that normalize arguments before calling a shared
  implementation
- quantized or nested helper schemas bound to upstream generic helpers

Do not treat missing local files for these wrappers as automatic Goal 1 or Goal
3 evidence.

## Source Layout Notes

Typical `torch-xpu-ops` structure:

- `yaml/native/native_functions.yaml` for authoritative backend dispatch and
  codegen overrides
- `src/ATen/native/xpu/` for XPU glue and host-side integration
- `src/ATen/native/xpu/sycl/` for SYCL kernel implementations
- `test/xpu/expect/HasDecompTest.test_has_decomposition.expect` as negative
  decomposition evidence only

Presence in `HasDecompTest.test_has_decomposition.expect` means decomposition is
still missing. Absence is not proof of support.

## Non-Bugs To Ignore During Dispatch Review

- SYCL versus CUDA synchronization API spelling
- oneDNN or oneMKL versus CUDA vendor library choices
- shorter or merged XPU kernels when semantics are equivalent
- lack of explicit XPU registration on a frontend wrapper that inherits a valid
  target path
- missing dedicated XPU file when the real implementation lives in upstream XPU
  or generic native code