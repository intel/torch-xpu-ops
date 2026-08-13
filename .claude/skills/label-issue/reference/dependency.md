# Dependency Rules

## Decision rules

- Return one taxonomy value, `none`, or `null`.
- Require direct traceback, source, or operator evidence. Missing or ambiguous
  evidence returns `null`; evidence that directly establishes no dependency
  returns `none`.
- Runtime and build failures are distinct. Do not infer a build dependency
  from a runtime failure or a runtime dependency from a build failure.
- For oneMKL or oneDNN classification, read `xpu_operator_dependency_list.md`
  in this same reference directory.

## Taxonomy and evidence mapping

- `driver`: version gate, submit/launch SYCL exception, or explicit driver.
- `IGC`: JIT, IGC, ocloc, or illegal instruction.
- `Level_Zero`: `zeXxx`, Level Zero, or enumeration evidence.
- `oneMKL`: operator listed in `xpu_operator_dependency_list.md`
  section 1.1, with its stated Condition confirmed. Complex-dtype matmul maps
  to `BlasImpl.cpp`; LU or triangular solve maps to `BatchLinearAlgebra.cpp`;
  FFT maps to `SpectralOps.cpp`.
- `oneDNN`: operator listed in section 1.2. Real-dtype matmul maps to
  `Blas.cpp`; convolution maps to `Conv.cpp`; fused linear maps to
  `Linear.cpp`; quantized maps to `qconv.cpp` / `qlinear.cpp`.
- `oneCCL`: ProcessGroupXCCL, c10d, or collective evidence.
- `oneAPI`: host DPC++ or `icpx` build only.
- `MSVC`: Windows C####, `cl.exe`, or `LINK.exe`.
- `Triton`: XPU triton or libtriton, or an Inductor Triton crash, but not an
  eager failure.
- `community`: a confirmed relevant OPEN `pytorch/pytorch` issue.
- `third_party_packages`: an originating `site-packages/<pkg>/` frame.

## Edge cases

- The dependency must be the direct cause, not incidental issue prose.
- If multiple taxonomy values remain equally supported, return `null`.
- Normalization ops (`batch_norm*`, `native_layer_norm`, `native_group_norm`),
  activations, and elementwise/reduction ops are native SYCL. They are never
  `oneMKL` or `oneDNN`. See section 1.3 of the operator lookup.
- An operator being listed under `oneMKL` or `oneDNN` is not sufficient. The
  failure must originate in that library, and any Condition in section 1.1 and
  the flip conditions in section 2 must be confirmed from evidence. Unconfirmed
  returns `null`.
