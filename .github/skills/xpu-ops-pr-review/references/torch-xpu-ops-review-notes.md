# Torch XPU Ops Review Notes

Use this file as the repository-specific overlay for `torch-xpu-ops` reviews.

## Why This Review Is Different

Do not treat `torch-xpu-ops` like a generic operator repository.

- Many operators are implemented here as SYCL kernels
- Some linear algebra paths route through oneMKL
- Some conv or gemm critical paths belong in upstream PyTorch oneDNN XPU code instead
- The first review question is whether the change belongs in this repository and this backend path at all

## Repository File Map

Inspect these first when the change touches operator wiring, kernels, or tests:

- `yaml/xpu_functions.yaml`
- `yaml/native/native_functions.yaml`
- `src/ATen/native/xpu/`
- `src/ATen/native/xpu/sycl/`
- `src/ATen/native/xpu/XPUFallback.template`
- `test/xpu/`
- `test/test_ops_xpu.py`
- `.github/workflows/` when the PR changes CI, packaging, or platform coverage

## Upstream Parity Source Of Truth

When CPU or CUDA parity is relevant, do not infer behavior from memory.

- Use an actual `pytorch/pytorch` source checkout as the reference for CPU and CUDA behavior
- If the workspace already has a local PyTorch checkout, inspect that code directly
- If no local PyTorch checkout is available, fetch or clone `pytorch/pytorch` before making parity claims
- Do not accept or write a CPU/CUDA parity conclusion unless the upstream implementation was actually inspected

## Eight High-Risk Review Areas

### 1. Semantic Parity Must Match CPU Or CUDA Behavior

Check:

- Invalid-input error behavior
- Dtype promotion and broadcasting
- Stride semantics and non-contiguous handling
- `functional`, `out=`, `inplace`, and view behavior
- Empty tensors, zero-size dimensions, scalar tensors, and channels-last inputs
- Backward, autograd, and deterministic behavior when relevant

Red flags:

- Only the happy path is tested
- Forward passes but backward or `out=` semantics are not covered
- The PR adds an XPU-only special case that silently changes PyTorch-visible behavior

### 2. Async Execution And Hidden Synchronization

Check:

- Host-side reads, `.item()`, debug printing, blocking helper APIs, or shape logic that depends on device values
- Cross-stream ordering when results are consumed elsewhere
- New `synchronize()` or wait calls added to force correctness
- Event or stream recording behavior when buffers or outputs live across streams

Red flags:

- A bug fix works only because it adds broad synchronization
- Device results are pulled back to host in the hot path
- Stream dependencies are implied but never expressed

### 3. Memory Layout And Format Handling

Check:

- Whether contiguous and non-contiguous inputs are both truly supported
- Whether channels-last inputs run a real optimized path or silently materialize contiguous intermediates
- Whether the change adds extra copies or format conversions
- Whether temporary buffers or `out=` allocations inflate memory usage or fragment the allocator

Red flags:

- The PR claims channels-last support but internally converts to plain contiguous format
- The implementation only works correctly after a hidden `.contiguous()`
- A temporary allocation is introduced for every launch without justification

### 4. Dtype, Precision, And Numerical Stability

Check:

- Input dtype, compute dtype, and accumulation dtype separately
- BF16, FP16, autocast, reductions, norms, softmax-style ops, and atomic accumulation patterns
- Whether a new optimization downgrades accumulation precision
- Whether tolerances are chosen per dtype rather than copied from CUDA defaults

Red flags:

- FP32 accumulation becomes BF16 or FP16 without explicit rationale
- Only FP32 tests exist for an op that is expected to run under BF16 or FP16
- Numerical thresholds do not match the dtype under test

### 5. 64-Bit Indexing And Large Tensor Safety

Check:

- Index, offset, stride, `numel`, and shape-product math
- Pointer arithmetic, flattened index helpers, and loop counters
- Whether address calculations can overflow even when total element count stays below `2^31`
- Whether tests cover realistic large-index cases

Red flags:

- `int` or `int32_t` is used for address math in generic tensor loops
- The PR passes small tests but has no evidence for large strides or offsets
- A helper assumes contiguous dense indexing and silently truncates

### 6. SYCL Kernel Mapping And Intel XPU Performance Model

Check:

- Work-group size and subgroup assumptions
- Branch-heavy inner loops or divergence
- Global-memory access patterns and repeated host-side setup
- Queue, context, descriptor, or temporary object construction inside hot paths
- Whether a generalized implementation slows down the dominant fast path

Red flags:

- Work-group size looks arbitrary with no test or benchmark evidence
- Every call rebuilds descriptors or other expensive launch metadata
- Rare edge-case support makes the common path branchy and slow

### 7. Dispatch, Fallback, And Registration

Check:

- Dispatch key registration and yaml wiring
- Meta, native, backward, and generated-code alignment
- Whether unsupported cases fail explicitly or intentionally fall back
- Whether the implementation location matches the registration path and backend boundary

Red flags:

- The PR updates a kernel file but not the yaml or generated wiring that should move with it
- A supposedly unsupported case silently takes a wrong generic path
- A library-backed path is reimplemented locally without a strong reason

### 8. Test Design Must Match XPU Risk, Not Just Functionality

At minimum, look for coverage across these dimensions when relevant:

1. Device dimension: XPU execution path is actually exercised
2. Dtype dimension: FP32, BF16, FP16, and integer types when supported
3. Layout dimension: contiguous, non-contiguous, and channels-last
4. Shape dimension: small, large, corner, empty, and broadcasted cases
5. API dimension: `functional`, `out=`, `inplace`, and backward
6. Execution dimension: default stream and non-default stream when async behavior matters
7. Numerical dimension: tolerances chosen per dtype
8. Performance dimension: benchmark or regression evidence when the PR claims optimization

Red flags:

- Only one tiny happy-path tensor shape is tested
- The tests cover CPU or generic wrappers but not the real XPU path
- A performance PR has no benchmark or regression evidence

## Practical Review Checklist

### A. Backend Ownership

- Should this change live in `torch-xpu-ops`, upstream PyTorch XPU, oneDNN XPU, or oneMKL?
- Is this duplicating an existing backend or library path?

### B. Correctness

- Does behavior match CPU or CUDA semantics?
- Are edge cases, `out=`, `inplace`, and backward covered?

### C. Async And Synchronization

- Is there hidden host synchronization?
- Are stream dependencies explicit and correct?
- Is synchronization being used to hide a race?

### D. Layout And Memory

- Is non-contiguous support real or just a hidden copy?
- Is channels-last truly optimized or only accepted?
- Are temporary buffers reasonable?

### E. Precision And Numerics

- Are BF16, FP16, autocast, and accumulation dtype handled correctly?
- Are tolerances chosen for XPU and dtype rather than copied blindly?

### F. Large Tensors

- Is indexing 64-bit safe?
- Can stride or offset math overflow?

### G. Performance

- Is the kernel mapping consistent with Intel XPU execution characteristics?
- Is there benchmark evidence for claimed optimization?

### H. Maintainability

- Are fast path and fallback path clearly separated?
- Are XPU-specific constraints explained where they are not obvious?
