# XPU Operator Dependency Lookup

Decides the `dependency` axis for operator-level failures: `oneMKL`, `oneDNN`,
`none` (native SYCL / CPU fallback), or `null` (insufficient evidence).

## Path conventions

All paths below are relative to one of two repository roots:

| Shorthand | Meaning |
|---|---|
| `<pt>/` | the PyTorch checkout, i.e. the skill's `pytorch_folder` input |
| `<xpu-ops>/` | `<pt>/third_party/torch-xpu-ops`, or a standalone torch-xpu-ops checkout |

When no `pytorch_folder` was provided, the shell commands in sections 3 and 4
cannot be run. Decide from sections 1 and 2 alone, and return `null` when the
required evidence is unavailable.

## How to use this file

1. Get the failing operator name from the traceback or test id.
2. Look it up in section 1. Exact overload first, then the base name.
3. If a **Condition** is listed, you must confirm it from issue evidence.
   Cannot confirm -> `null`, not a guess.
4. Not in any table -> section 3.

An operator appearing here is **not** by itself evidence of a dependency. The
failure must actually originate in that library.

## 1. Operator lookup

### 1.1 oneMKL

Source: `<xpu-ops>/src/ATen/native/xpu/mkl/`. All rows also require the build to
have `USE_ONEMKL_XPU` on (default on; see section 2).

| Operator | Condition | File |
|---|---|---|
| `mm`, `mm.out` | **complex dtype only** | `BlasImpl.cpp` |
| `bmm`, `bmm.out` | **complex dtype only** | `BlasImpl.cpp` |
| `addmm`, `addmm.out` | **complex dtype only** | `BlasImpl.cpp` |
| `baddbmm`, `baddbmm.out` | **complex dtype only** | `BlasImpl.cpp` |
| `dot` | none | `BlasImpl.cpp` |
| `vdot` | none | `BlasImpl.cpp` |
| `_fft_c2c` (+`.out`) | not on SYCL FFT path | `SpectralOps.cpp` |
| `_fft_c2r` (+`.out`) | none | `SpectralOps.cpp` |
| `_fft_r2c` (+`.out`) | none | `SpectralOps.cpp` |
| `linalg_lu_factor_ex.out` | **batch > 1** | `BatchLinearAlgebra.cpp` |
| `linalg_lu_solve.out` | none | `BatchLinearAlgebra.cpp` |
| `linalg_solve_triangular` (+`.out`) | none | `BatchLinearAlgebra.cpp` |
| `orgqr`, `linalg_householder_product` | none | `BatchLinearAlgebra.cpp` |

Real-dtype `mm`/`bmm`/`addmm`/`baddbmm` are **oneDNN**, not oneMKL.

### 1.2 oneDNN

Source: `<pt>/aten/src/ATen/native/mkldnn/xpu/`.

| Operator | File |
|---|---|
| `mm`, `bmm`, `addmm`, `baddbmm` (+`.out`, `.dtype`, `.dtype_out`) - real dtypes | `Blas.cpp` |
| `addmv.out` | `Blas.cpp` |
| `_addmm_activation.out` | `Blas.cpp` |
| `_int_mm` (+`.out`) | `Blas.cpp` |
| `_weight_int4pack_mm`, `_weight_int8pack_mm` | `Blas.cpp` |
| `_scaled_mm` (+`.out`), `_scaled_mm_v2.out` | `ScaledBlas.cpp` |
| `convolution_overrideable` (+`_backward`) - covers `conv1d/2d/3d`, `conv_transpose*d` | `Conv.cpp` |
| `mkldnn::_convolution_pointwise*` (Inductor fusion) | `Conv.cpp` |
| `mkldnn::_linear_pointwise*` (Inductor fusion) | `Linear.cpp` |
| `onednn::qconv*`, `onednn::qlinear*` (quantized) | `qconv.cpp`, `qlinear.cpp` |
| LSTM inference (`lstm_mkldnn_stub`) | `RNN.cpp` |
| `_scaled_dot_product_fused_attention_overrideable` | `Attention.cpp` |

`linear` and `conv2d`/`conv3d` are Composite: they carry no XPU registration and
decompose into `addmm`/`mm` and `convolution_overrideable` above. Attribute a
`linear` or `conv` failure to the operator it decomposed into.

### 1.3 NOT a dependency - commonly misattributed

These have no oneMKL or oneDNN code path. A failure here is `none`, not `oneDNN`.

| Operator | Actual backing |
|---|---|
| `batch_norm`, `native_batch_norm`, `_batch_norm_with_update`, `batch_norm_stats`, `batch_norm_elemt`, `batch_norm_backward*`, `batch_norm_update_stats` | SYCL `BatchNormKernels.cpp` |
| `native_layer_norm` (+`_backward`) | SYCL `LayerNormKernels.cpp` |
| `native_group_norm` (+`_backward`) | SYCL `GroupNormKernels.cpp` |
| `softmax`, `_log_softmax` | SYCL |
| all activations (`relu`, `gelu`, `silu`, ...) | SYCL |
| all elementwise / reduction / indexing / sorting ops | SYCL |
| `geqrf`, `geqrf.a` | CPU fallback (never oneMKL) |

Verify: `grep -niE "onednn|mkldnn|dnnl" <file>` under
`<xpu-ops>/src/ATen/native/xpu/` prints nothing for all of the above.

### 1.4 CPU fallback -> `none`

Always fall back to CPU regardless of environment, per the `fallback_list` in
`<xpu-ops>/src/ATen/native/xpu/XPUFallback.cpp`. No XPU kernel exists, so there
is no library dependency:

```
cholesky_inverse                 linalg_householder_product
cholesky_inverse.out             linalg_householder_product.out
_cholesky_solve_helper           linalg_ldl_factor_ex.out
_efficient_attention_forward     linalg_ldl_solve.out
geqrf                            linalg_lstsq.out
geqrf.a                          linalg_lu.out
hash_tensor.out                  linalg_matrix_exp
linalg_cholesky_ex.L             linalg_matrix_sqrth
linalg_eig                       linalg_polar.out
linalg_eig.out                   linalg_qr.out
_linalg_eigvals                  _linalg_svd.U
linalg_eigvals.out               lu_unpack.out
_linalg_eigh.eigenvalues         ormqr
triangular_solve.X               ormqr.out
_validate_compressed_sparse_indices
```

Note `linalg_householder_product` and `orgqr` appear in **both** 1.1 and here:
the stub reaches oneMKL when built with it, else CPU. Requires evidence to split.

## 2. Conditions that flip the answer

Check each before returning `oneMKL` or `oneDNN`.

| Condition | Effect | Evidence to look for |
|---|---|---|
| dtype | complex -> oneMKL; real -> oneDNN | dtype in test id / repro / error |
| `USE_ONEMKL_XPU` off | oneMKL rows become CPU fallback -> `none` | log line `Consider building with USE_ONEMKL_XPU=1` |
| `USE_SYCL_SPECTRAL=1` | FFT goes to SYCL -> `none` | env in issue body |
| batch size 1 | `linalg_lu_factor_ex.out` goes to CPU -> `none` | tensor shape in repro |
| `PYTORCH_XPU_FALLBACK_OP` set | named overloads forced to CPU -> `none` | env in issue body |
| `PYTORCH_ENABLE_XPU_FALLBACK=1` | unregistered ops silently run on CPU | env in issue body |

Warning: the `mm` family crosses repos. YAML dispatches `mm.out` to
`mm_out_xpu` in the PyTorch **oneDNN** file, which delegates complex inputs to
torch-xpu-ops `mm_complex_out_xpu` -> oneMKL. A complex-dtype `mm` bug is
**oneMKL / torch-xpu-ops** despite the oneDNN entry point.

## 3. Operator not in any table

Resolve it, do not guess. Dependency is not recorded in the YAML - it only
gives a kernel symbol. Requires `pytorch_folder`; without it, return `null`.

```bash
cd "$pytorch_folder"

# 1. find the XPU symbol (use a wide window; XPU: sits 7+ lines below func:,
#    and SparseXPU / SparseCsrXPU / NestedTensorXPU are DIFFERENT kernels)
grep -n -A12 '^- func: <op>' \
  aten/src/ATen/native/native_functions.yaml | grep -E 'func:|XPU'

# 2. locate that symbol
grep -rn "<symbol>" aten/src/ATen/native/mkldnn/xpu/ third_party/torch-xpu-ops/src/
```

| Symbol lands in | Verdict |
|---|---|
| `<pt>/aten/src/ATen/native/mkldnn/xpu/` | `oneDNN` (fix in pytorch) |
| `<xpu-ops>/src/ATen/native/xpu/mkl/` | `oneMKL` (fix in torch-xpu-ops) |
| `<xpu-ops>/src/ATen/native/xpu/sycl/` | `none` |
| `<xpu-ops>/src/ATen/native/xpu/*.cpp` | read the body; it may branch on dtype / `USE_ONEMKL_XPU` / batch size |
| nowhere; no XPU entry | Composite, or unimplemented -> catch-all fallback -> `none` |

No local checkout, or the symbol cannot be resolved -> `null`.

## 4. Quick verification

Run from a PyTorch checkout to refresh this file after a pull.

```bash
cd "$pytorch_folder"
XPUOPS=third_party/torch-xpu-ops

# oneMKL surface (3 files) and every file gated on it
ls $XPUOPS/src/ATen/native/xpu/mkl/
grep -rl "USE_ONEMKL_XPU" $XPUOPS/src/

# oneDNN surface (9 files) and its registrations
grep -nE "TORCH_LIBRARY_IMPL|REGISTER_XPU_DISPATCH" \
  aten/src/ATen/native/mkldnn/xpu/*.cpp

# current fallback list (section 1.4 drifts often)
awk '/std::vector<std::string> fallback_list/,/};/' \
  $XPUOPS/src/ATen/native/xpu/XPUFallback.cpp

# both MUST print nothing: norms are SYCL, and oneDNN is not gated in torch-xpu-ops
grep -niE "onednn|mkldnn|dnnl" \
  $XPUOPS/src/ATen/native/xpu/{BatchNorm,LayerNorm,GroupNorm}.cpp
grep -rniE "onednn|mkldnn" $XPUOPS/cmake/
```

Re-check sections 1.2 and 1.4 after pulling either repo; both change often.
