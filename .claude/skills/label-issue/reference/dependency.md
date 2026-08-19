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

This axis is decided BEFORE ownership, and its result drives it: a taxonomy
value becomes the `target_component` verbatim and forces `need_action:
NEED_FIX_3RDPARTY` in [target_component.md](target_component.md). So
`dependency: oneDNN` pairs with `target_component: oneDNN`, not with a generic
`third-party`. `none` and `null` force nothing. Because a taxonomy value both
overrides the traced fix location and names the owner, return one only on direct
evidence that the failure originates in that component — never on a library
merely appearing in the call path.

`extract.json` carries the value in `dependency` and its label in
`dependency_label`, both read off the issue's EXISTING dependency label. Treat
them as a prior, not an answer: still decide this axis from the rules above, and
when you override the value take its label from the mapping table below. Both
fields are `""` when the issue carries no dependency label — that is not
evidence of `none`. Emit the label, never the bare value.

## Value to label mapping

Emit the **label** column in `labels.md`, never the bare value. Note that
`third_party_packages` uses a different prefix and a space, so the
`dependency component: <value>` pattern does NOT hold for it.

| Value | GitHub label | Label exists today |
|---|---|---|
| `driver` | `dependency component: driver` | yes |
| `oneDNN` | `dependency component: oneDNN` | yes |
| `oneMKL` | `dependency component: oneMKL` | yes |
| `oneAPI` | `dependency component: oneAPI` | yes |
| `Triton` | `dependency component: Triton` | yes |
| `MSVC` | `dependency component: MSVC` | yes |
| `community` | `dependency component: community` | yes |
| `third_party_packages` | `dependency: third_party packages` | yes |
| `oneCCL` | `dependency component: oneCCL` | **no - must be created** |
| `IGC` | `dependency component: IGC` | **no - must be created** |
| `Level_Zero` | `dependency component: Level_Zero` | **no - must be created** |

The last three do not exist in `intel/torch-xpu-ops` yet. `labels.md` is a
proposal, so emit them anyway and note in the reason that the human must create
the label before applying it.

`AO` is NOT a dependency value. torchao is a PyTorch-ecosystem component owned by
the module axis (`module: ao`), not an external dependency. A transformers or
huggingface failure is `third_party_packages`; `module: transformers` may carry
the domain signal separately.

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
- `third_party_packages`: an originating non-torch `site-packages/<pkg>/` frame,
  such as `site-packages/transformers/`. A `site-packages/torch/` frame is
  PyTorch itself and is NOT this value.

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
- Ignore the `## Versions` / `Collecting environment` dump when keyword matching.
  It lists `onemkl`, `oneccl` and `intel-sycl-rt` for every issue, so exclude
  that section from consideration.
