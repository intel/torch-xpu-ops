# Dependency Rules

## Decision rules

- Return one taxonomy value, `none`, or `null`.
- Require direct traceback, source, or operator evidence. Missing or ambiguous
  evidence returns `null`; evidence that directly establishes no dependency
  returns `none`.
- Runtime and build failures are distinct. Do not infer a build dependency
  from a runtime failure or a runtime dependency from a build failure.
- For oneMKL or oneDNN classification, read
  `.opencode/skills/validation/issue-triage/reference/xpu_supported_operators_complete_list.md`.

## Taxonomy and evidence mapping

- `driver`: version gate, submit/launch SYCL exception, or explicit driver.
- `IGC`: JIT, IGC, ocloc, or illegal instruction.
- `Level_Zero`: `zeXxx`, Level Zero, or enumeration evidence.
- `oneMKL`: Part I section 1.2. BLAS/matmul maps to `BlasImpl.cpp`; LU,
  Cholesky, or solve maps to `BatchLinearAlgebra.cpp`; FFT maps to
  `SpectralOps.cpp`.
- `oneDNN`: Part I section 1.3. Convolution maps to `Conv.cpp`; linear maps to
  `Linear.cpp`; batch norm maps to `BatchNorm.cpp`.
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
