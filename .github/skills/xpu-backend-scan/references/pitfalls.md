# Non-Obvious Pitfalls

Things the agent cannot reliably derive from reading code alone.

## Judgment Traps

- **Structured delegate**: `structured_delegate: foo.out` means support is judged by the `foo.out` target, not the wrapper. A missing XPU file for the wrapper is NOT evidence of missing support.
- **Wrapper inheritance**: Thin frontend shims that redispatch to `*.out` or normalize arguments do not need separate XPU registration.
- **No dedicated kernel file ≠ missing**: The op may use structured delegate, composite dispatch, or a shared path. Always check backend YAML and `structured_delegate` target.
- **No backend-local backward symbol ≠ missing backward**: Generic autograd or `derivatives.yaml` shared formulas may cover it.
- **CUDA entry in YAML without XPU ≠ XPU absence**: Check all coverage paths (native, composite, decomp, fallback) before concluding.

## Fallback Mechanisms

`XPUFallback.template` has three relevant fallback paths:
1. **Hardcoded per-op fallback list** — selected ops are wired to `xpu_fallback_impl` (CPU).
2. **Env-forced per-op fallback** — `PYTORCH_XPU_FALLBACK_OP` can force fallback for named ops at runtime.
3. **Global backend fallback** — default is `TORCH_CHECK_NOT_IMPLEMENTED`; becomes CPU fallback only when `PYTORCH_ENABLE_XPU_FALLBACK=1`.

All indicate the op lacks a native XPU kernel for that path. Report as "fallback only" (low priority) unless other evidence shows native coverage.

## Snapshot File

`HasDecompTest.test_has_decomposition.expect` is a **negative** snapshot — if an op appears in this file, that indicates its decomposition is *missing*, not present.
