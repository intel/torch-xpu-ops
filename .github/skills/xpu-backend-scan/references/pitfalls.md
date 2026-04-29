# Non-Obvious Pitfalls

Things the agent cannot reliably derive from reading code alone.

## Judgment Traps

- **Structured delegate**: `structured_delegate: foo.out` means support is judged by the `foo.out` target, not the wrapper. A missing XPU file for the wrapper is NOT evidence of missing support.
- **Wrapper inheritance**: Thin frontend shims that redispatch to `*.out` or normalize arguments do not need separate XPU registration.
- **No dedicated kernel file ≠ missing**: The op may use structured delegate, composite dispatch, or a shared path. Always check backend YAML and `structured_delegate` target.
- **No backend-local backward symbol ≠ missing backward**: Generic autograd or `derivatives.yaml` shared formulas may cover it.
- **CUDA entry in YAML without XPU ≠ XPU absence**: Check all coverage paths (native, composite, decomp, fallback) before concluding.

## Fallback Mechanisms

`XPUFallback.template` has two distinct mechanisms:
1. **Explicit per-op fallback** — always routes to CPU via `xpu_fallback_impl`.
2. **Global backend fallback** — default is `TORCH_CHECK_NOT_IMPLEMENTED`; becomes CPU fallback only when `PYTORCH_ENABLE_XPU_FALLBACK=1`.

Both mean XPU lacks a native GPU kernel. Report as "fallback only" (low priority).

## Snapshot File

`HasDecompTest.test_has_decomposition.expect` is a **negative** snapshot — presence means decomposition is *missing*, not present.
