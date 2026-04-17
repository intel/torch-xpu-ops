# XPU Ops Review Checklist

Use this checklist for `torch-xpu-ops` PR reviews. It is intentionally focused on XPU-specific correctness, backend wiring, and CI-blind risks rather than style or lint.

## Scope And Ownership

- [ ] The change belongs in `torch-xpu-ops` rather than upstream PyTorch XPU, oneDNN XPU, or oneMKL
- [ ] The implementation location matches the intended backend path and registration strategy
- [ ] The PR does not duplicate an existing library or backend capability without justification

## Correctness And Semantics

- [ ] CPU or CUDA parity claims were checked against the exact counterpart
- [ ] Input validation, output semantics, dtype promotion, and broadcasting behavior are still correct
- [ ] Empty tensors, zero-size dims, scalar tensors, and non-contiguous cases are handled or rejected intentionally
- [ ] `functional`, `out=`, `inplace`, view, and backward behavior still line up when relevant

## Async Execution And Synchronization

- [ ] No hidden host synchronization was introduced in a hot or common path
- [ ] Cross-stream ordering is explicit where the change relies on it
- [ ] New waits or synchronizations are required for correctness rather than masking a race
- [ ] The PR does not pull device results back to host just to drive control flow or debugging

## Layout, Memory Format, And Allocation

- [ ] Non-contiguous support is real rather than a silent `.contiguous()` fallback
- [ ] Channels-last support is either genuinely optimized or explicitly treated as a conversion path
- [ ] Output and temporary allocation choices preserve the intended memory format when needed
- [ ] Extra copies, format conversions, or oversized temporaries are justified

## Dtype, Precision, And Numerics

- [ ] Compute dtype and accumulation dtype are appropriate for FP32, BF16, FP16, and integer paths
- [ ] Reductions, norms, softmax-style kernels, and atomic accumulation patterns were checked for numerical stability
- [ ] Autocast behavior is covered when the operator participates in mixed precision
- [ ] Test tolerances are chosen per dtype rather than copied blindly from another backend

## Large Tensor Safety

- [ ] Index, offset, stride, and `numel` math are 64-bit safe where required
- [ ] Pointer arithmetic and flattened indexing helpers cannot overflow silently
- [ ] Large-stride or large-offset behavior has evidence beyond tiny tensors

## Kernel Mapping And Performance

- [ ] Work-group or subgroup decisions look deliberate rather than arbitrary
- [ ] The hot path is not made branch-heavy just to support rare cases
- [ ] Expensive queue, context, descriptor, or host-side setup is not repeated unnecessarily
- [ ] Claimed optimizations come with benchmark evidence or at least a concrete design rationale

## Dispatch, Fallback, And Generated Wiring

- [ ] Relevant yaml, native implementation, backward logic, and generated expectations move together
- [ ] Unsupported cases fail explicitly or intentionally fall back rather than taking the wrong path silently
- [ ] The XPU path exercised by tests is the same path wired by the registration

## Test Coverage

- [ ] Tests exercise the actual XPU path rather than only generic wrappers
- [ ] Coverage exists across relevant dtypes, layouts, shapes, and API variants
- [ ] Async-risk changes include stream-sensitive tests when appropriate
- [ ] Performance PRs include benchmark or regression evidence

## Backward Compatibility

- [ ] Any user-visible behavior change is called out explicitly
- [ ] Error type, default behavior, output semantics, and determinism changes were reviewed as BC-sensitive
- [ ] If the change intentionally alters public behavior, the rationale is stronger than "tests pass"