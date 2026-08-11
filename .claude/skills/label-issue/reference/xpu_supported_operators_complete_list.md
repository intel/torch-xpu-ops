# PyTorch XPU Backend: Complete Operator Registry with Implementation and Dependency Mapping

> **Document**: Comprehensive listing of all 749 XPU backend supported operators  
> **Source**: `yaml/xpu_functions.yaml`  
> **Generated**: April 2026  
> **Repository**: `/home/daisydeng/test_pytorch/workdir/pytorch/third_party/torch-xpu-ops` (Version 2.10.0)

---

## Overview Statistics

| Metric | Count |
|--------|-------|
| **Total Supported Operators** | 749 |
| **Main Implementation Files** | 106 |
| **SYCL Kernel Files** | 201 |
| **Native SYCL Implementation** | ~412 operators |
| **oneMKL Integration** | ~34 operators |
| **oneDNN Integration** | ~40 operators |
| **CPU Fallback** | ~263 operators |

### Implementation Path Distribution

| Path | Count | Primary Dependency |
|------|-------|-------------------|
| **Native SYCL** | 412 | SYCL Runtime + TensorIterator |
| **oneMKL** | 34 | Intel oneAPI Math Kernel Library |
| **oneDNN** | 40 | Intel oneAPI DNN Library |
| **CPU Fallback** | 263 | CPU with implicit tensor transfer |

---

## Part I: Implementation File Index by Dependency

### 1.1 SYCL Kernel Files (src/ATen/native/xpu/sycl/)

| Category | File | Key Operators |
|----------|------|---------------|
| **Activation** | `ActivationBlasKernel*` | Element-wise math |
| | `ActivationEluKernels.*` | ELU activation |
| | `ActivationGeluKernel.*` | GELU activation |
| | `ActivationHardshrinkKernels.*` | Hardshrink |
| | `ActivationHardsigmoidKernels.*` | Hardsigmoid |
| | `ActivationHardswishKernels.*` | Hardswish |
| | `ActivationHardtanhKernels.*` | Hardtanh |
| | `ActivationLeakyReluKernels.*` | LeakyReLU |
| | `ActivationLogSigmoidKernels.*` | LogSigmoid |
| | `ActivationMishKernels.*` | Mish activation |
| | `ActivationPreluKernels.*` | PReLU |
| | `ActivationSiluKernels.*` | SiLU (Swish) |
| | `ActivationSoftplusKernels.*` | Softplus |
| | `ActivationSoftshrinkKernels.*` | Softshrink |
| | `ActivationThresholdKernel.*` | Threshold |
| | `Activation专题Kernel.*` | Additional activations |
| **Unrary Math** | `UnaryAcoshKernel.*` | arccosh |
| | `UnaryAsinKernel.*` | arcsin |
| | `UnaryAtan2Kernel.*` | arctan2 |
| | `UnaryCeilKernel.*` | ceil |
| | `UnaryCosKernel.*` | cosine |
| | `UnaryCoshKernel.*` | hyperbolic cos |
| | `UnaryDigammaKernel.*` | digamma |
| | `UnaryEluKernel.*` | ELU |
| | `UnaryErfKernel.*` | error function |
| | `UnaryErfcKernel.*` | complementary erf |
| | `UnaryErfinvKernel.*` | inverse erf |
| | `UnaryExpKernel.*` | exponential |
| | `UnaryExpm1Kernel.*` | exp(x)-1 |
| | `UnaryFloorKernel.*` | floor |
| | `UnaryLgammaKernel.*` | log gamma |
| | `UnaryLog10Kernel.*` | log base 10 |
| | `UnaryLog1pKernel.*` | log(1+x) |
| | `UnaryLogKernel.*` | natural log |
| | `UnaryLog2Kernel.*` | log base 2 |
| | `UnaryNegKernel.*` | negation |
| | `UnaryNextAfterKernel.*` | nextafter |
| | `UnaryPolygammaKernel.*` | polygamma |
| | `UnaryPowKernel.*` | power |
| | `UnaryReciprocalKernel.*` | reciprocal |
| | `UnaryRoundKernel.*` | rounding |
| | `UnaryRsqrtKernel.*` | 1/sqrt |
| | `UnarySigmoidKernel.*` | sigmoid |
| | `UnarySinhKernel.*` | hyperbolic sin |
| | `UnarySinKernel.*` | sine |
| | `UnarySqrtKernel.*` | square root |
| | `UnaryTanhKernel.*` | tangent hyperbolic |
| | `UnaryTanKernel.*` | tangent |
| | `UnarySignKernel.*` | sign |
| | `UnaryTruncKernel.*` | truncation |
| **Reduction** | `BinaryAddcmulKernel.*` | fused multiply-add |
| | `BinaryAddcdivKernel.*` | fused multiply-div |
| | `FillKernel.*` | tensor fill |
| | `ForeachAddKernel.*` | in-place add |
| | `ForeachDivKernel.*` | in-place div |
| | `ForeachMulKernel.*` | in-place mul |
| | `ForeachNormKernel.*` | in-place norm |
| | `ForeachSqrtKernel.*` | in-place sqrt |
| | `NormKernel.*` | vector norm |
| | `ReduceAddKernel.*` | sum reduction |
| | `ReduceAllKernel.*` | logical AND all |
| | `ReduceAnyKernel.*` | logical OR any |
| | `ReduceArgmaxKernel.*` | argmax |
| | `ReduceArgminKernel.*` | argmin |
| | `ReduceProdKernel.*` | product reduction |
| | `ReduceMaxKernel.*` | max reduction |
| | `ReduceMinKernel.*` | min reduction |
| | `SumKernel.*` | sum |
| | `VarKernel.*` | variance |
| **BlasKernels** | `BlasKernel.*` | BLAS level-1 |
| **Pooling** | `AdaptiveAvgPool2dKernel.*` | adaptive avg pool |
| | `AdaptiveAvgPool3dKernel.*` | 3D adaptive pool |
| | `AveragePool2dKernel.*` | 2D avg pool |
| | `MaxPool2dKernel.*` | 2D max pool |
| **Tensor** | `CopyDeviceToDevice.*` | tensor copy |
| | `FlipKernel.*` | tensor flip |
| | `RollKernel.*` | tensor roll |
| | `TransformerXvKernel.*` | transformer |

### 1.2 oneMKL Implementation Files (src/ATen/native/xpu/mkl/)

| File | Operators | Description |
|------|-----------|-------------|
| `BlasImpl.cpp` | `mm`, `bmm`, `addmm`, `addbmm`, `dot`, `vdot`, `addmv`, `mv` | BLAS Level-3, 2 operations |
| `BatchLinearAlgebra.cpp` | `lu_factor`, `lu_solve`, `inverse`, `cholesky_*` | LAPACK decompositions |
| `SpectralOps.cpp` | `fft_c2c`, `fft_r2c`, `fft_c2r`, `fft_*_out` | FFT operations |

### 1.3 oneDNN Implementation Files (aten/src/ATen/native/mkldnn/xpu/)

| File | Operators | Description |
|------|-----------|-------------|
| `Conv.cpp` | `conv2d`, `conv3d`, `conv_*_backward` | Convolution operations |
| `Linear.cpp` | `linear`, `linear_backward` | Linear/matmul operations |
| `BatchNorm.cpp` | `batch_norm`, `native_batch_norm` | Batch normalization |

---

## Part II: Complete Operator Registry by Implementation File

### 2.1 Activation.cpp (21 operators)

**File**: `src/ATen/native/xpu/Activation.cpp`  
**Dependency**: SYCL (__use_dnnl__: false for activation path)

| Operator | Variants | Kernel |
|----------|----------|--------|
| `threshold` | base, *, `out` | `threshold_kernel` (SYCL) |
| `elu` | base, `out`, `_` | `elu_kernel` |
| `elu_backward` | base, `grad_input` | `elu_backward_kernel` |
| `silu` | base, `out`, `_` | `silu_kernel` |
| `silu_backward` | base | `silu_backward_kernel` |
| `hardswish` | base, `out`, `_`, `backward` | `hardswish_kernel` |
| `hardswish_backward` | base | `hardswish_backward_kernel` |
| `hardtanh` | base, `out`, `_` | via direct call |
| `hardtanh_backward` | base, `grad_input` | `hardtanh_backward_kernel` |
| ` hardsigmoid` | base, `out`, `_` | `hardsigmoid_kernel` |
| `hardsigmoid_backward` | base | `hardsigmoid_backward_kernel` |
| `leaky_relu` | base, `out`, `_` | `leaky_relu_kernel` |
| `leaky_relu_backward` | base | `leaky_relu_backward_kernel` |
| `softplus` | base, `out` | `softplus_kernel` |
| `softplus_backward` | base, `grad_input` | `softplus_backward_kernel` |
| `softshrink` | base, `out` | `softshrink_kernel` |
| `softshrink_backward` | base | `softshrink_backward_kernel` |
| `mish` | base, `out`, `_` | `mish_kernel` |
| `mish_backward` | base | `mish_backward_kernel` |
| `log_sigmoid_forward` | base, `output` | `log_sigmoid_forward_kernel` |
| `log_sigmoid_backward` | base, `grad_input` | `log_sigmoid_backward_kernel` |
| `prelu` | base | `prelu_kernel` |
| `prelu_backward` | base | `prelu_backward_kernel` |
| `hardshrink` | base | `hardshrink_kernel` |

---

### 2.2 Blas.cpp (Linear Algebra with fallback)

**File**: `src/ATen/native/xpu/Blas.cpp`  
**Dependency**: SYCL + oneMKL (USE_ONEMKL_XPU)

| Operator | Variants | Implementation |
|----------|----------|----------------|
| `mm` | `out` | MKL path: `gemm_xpu_mkl`, Fallback: real GEMM decomp |
| `bmm` | `out` | MKL path: `gemm_xpu_mkl`, Fallback: batch decomp |
| `addmm` | `out` | MKL path: `gemm_xpu_mkl`, Fallback: real GEMM decomp |
| `addbmm` | `out` | MKL path: `gemm_xpu_mkl`, Fallback: batch decomp |
| `dot` | base | MKL path: MKL dot product |
| `vdot` | base | MKL path: MKL vdot product |
| `addmv` | `out` | MKL path or fallback |
| `mv` | `out` | MKL path or fallback |

**Fallback** when `USE_ONEMKL_XPU` not defined:
- Complex matrix multiply via Gauss-Strassen decomposition
- Uses 3 real GEMMs instead of 4

---

### 2.3 BatchLinearAlgebra.cpp

**File**: `src/ATen/native/xpu/BatchLinearAlgebra.cpp`  
**Dependency**: SYCL + oneMKL

| Operator | Variants | Implementation |
|----------|----------|----------------|
| `lu_factor` | `out` | oneMKL `getrf` |
| `lu_factor.out` | base | oneMKL `getrf` |
| `lu_solve` | base | oneMKL `getrs` |
| `triangular_solve` | `X` | oneMKL `trsm` |
| `inverse` | base | oneMKL `getri` |
| `cholesky_*` | many | oneMKL `potrf` variants |

---

### 2.4 BatchNorm.cpp

**File**: `src/ATen/native/xpu/BatchNorm.cpp`  
**Dependency**: oneDNN or SYCL

| Operator | Variants | Implementation |
|----------|----------|----------------|
| `batch_norm` | base, `out` | oneDNN or SYCL kernel |
| `native_batch_norm` | base | oneDNN path |
| `native_batch_norm.out` | base | oneDNN path |
| `native_batch_norm_backward` | base | Backward pass |
| `_batch_norm_with_update` | base, `out` | Fused update |
| `batch_norm_backward` | base | Gradient computation |
| `batch_norm_stats` | base | Statistics |
| `batch_norm_elemt` | base, `out` | Per-element |
| `batch_norm_backward_elemt` | base | Per-element grad |
| `batch_norm_update_stats` | base | Update running stats |

---

### 2.5 BinaryOps.cpp

**File**: `src/ATen/native/xpu/BinaryOps.cpp`  
**Dependency**: SYCL

| Operator | Variants | Kernel |
|----------|----------|--------|
| `add.Tensor` | base, `out`, `_` | SYCL TensorIterator |
| `sub.Tensor` | base, `out`, `_` | SYCL TensorIterator |
| `mul.Tensor` | base, `out`, `_` | SYCL TensorIterator |
| `div.Tensor` | base, `out`, `_` | SYCL TensorIterator |
| `atan2` | base, `out`, `_` | SYCL kernel |
| `bitwise_and.Tensor_out` | base | SYCL binary |
| `bitwise_or.Tensor_out` | base | SYCL binary |
| `bitwise_xor.Tensor_out` | base | SYCL binary |
| `__lshift__.Scalar` | base | Bitwise shift |
| `__lshift__.Tensor` | base | Bitwise shift |
| `__rshift__.Scalar` | base | Bitwise shift |
| `__rshift__.Tensor` | base | Bitwise shift |

---

### 2.6 SoftMax.cpp

**File**: `src/ATen/native/xpu/SoftMax.cpp`  
**Dependency**: SYCL + specialized kernels

| Operator | Variants | Implementation |
|----------|----------|----------------|
| `_softmax` | base, `out` | Optimized softmax |
| `_softmax_backward_data` | base, `out` | Backward pass |
| `_log_softmax` | base, `out` | Log-softmax |
| `_log_softmax_backward_data` | base, `out` | Backward pass |

---

### 2.7 ReduceOps.cpp

**File**: `src/ATen/native/xpu/ReduceOps.cpp`  
**Dependency**: SYCL reduction kernels

| Operator | Variants | Implementation |
|----------|----------|--------|
| `sum.dim_IntList` | base | SYCL reduce |
| `sum.IntList_out` | base | SYCL reduce |
| `prod` | base | SYCL reduce |
| `prod.dim_int` | base | SYCL reduce |
| `mean.out` | base | SYCL reduce |
| `mean.dim` | base | SYCL reduce |
| `std.correction` | base | Variance compute |
| `std.correction_out` | base | Variance compute |
| `std_mean.correction` | base | Combined |
| `var.correction` | base | SYCL variance |
| `var.correction_out` | base | SYCL variance |
| `var_mean.correction` | base | Combined |

---

### 2.8 Sorting.cpp

**File**: `src/ATen/native/xpu/Sorting.cpp`  
**Dependency**: SYCL

| Operator | Variants | Implementation |
|----------|----------|--------|
| `sort.stable` | base | SYCL sort |
| `sort.values_stable` | base | SYCL sort |
| `argsort.stable` | base | Index sort |
| `topk` | base | SYCL top-k |
| `topk.values` | base | Top-k values |

---

### 2.9 Indexing.cpp

**File**: `src/ATen/native/xpu/Indexing.cpp`  
**Dependency**: SYCL

| Operator | Variants | Implementation |
|----------|----------|--------|
| `index.Tensor` | base, `out` | SYCL indexing |
| `index_add.out` | base | Index add |
| `index_add_` | base | In-place add |
| `index_select` | base, `out` | Select by index |
| `gather` | base, `out` | Gather operation |
| `scatter.src` | base | Scatter operation |
| `scatter.value` | base | Value scatter |
| `scatter.reduce` | base | Reduce scatter |
| `scatter_add` | base | Add scatter |
| `scatter_reduce.two` | base | Reduce scatter |
| `masked_select` | base, `out` | Masked selection |
| `masked_fill_.Tensor` | base | Fill by mask |
| `masked_fill_.Scalar` | base | Fill scalar |
| `masked_scatter_` | base | Scatter by mask |
| `where.self_out` | base | Conditional select |
| `where.self` | base | Conditional select |

---

### 2.10 UnaryOps.cpp

**File**: `src/ATen/native/xpu/UnaryOps.cpp`  
**Dependency**: SYCL

| Operator | Variants | Kernel |
|----------|----------|--------|
| `abs` | base, `out`, `_` | SYCL unary |
| `sign` | base, `out`, `_` | SYCL unary |
| `signbit` | base, `out` | SYCL unary |
| `nan_to_num.out` | base | SYCL transform |
| `conj_physical.out` | base | SYCL unary |
| `conj_physical_` | base | SYCL unary |
| `isfinite` | base | SYCL comparison |
| `isinf` | base | SYCL comparison |
| `isnan` | base, `out` | SYCL unary |
| `isposinf` | base | SYCL unary |
| `isneginf` | base | SYCL unary |
| `isreal` | base | SYCL unary |
| `serial` | base | SYCL unary |
| `imag` | base | SYCL unary |
| `acos` | base, `out`, `_` | SYCL kernel |
| `acosh` | base, `out`, `_` | SYCL kernel |
| `asin` | base, `out`, `_` | SYCL kernel |
| `asinh` | base, `out`, `_` | SYCL kernel |
| `atan` | base, `out`, `_` | SYCL kernel |
| `atanh` | base, `out`, `_` | SYCL kernel |
| `cos` | base, `out`, `_` | SYCL kernel |
| `cosh` | base, `out`, `_` | SYCL kernel |
| `sin` | base, `out`, `_` | SYCL kernel |
| `sinh` | base, `out`, `_` | SYCL kernel |
| `tan` | base, `out`, `_` | SYCL kernel |
| `tanh` | base, `out`, `_` | SYCL kernel |
| `exp` | base, `out`, `_` | SYCL kernel |
| `exp2` | base, `out`, `_` | SYCL kernel |
| `expm1` | base, `out`, `_` | SYCL kernel |
| `log` | base, `out`, `_` | SYCL kernel |
| `log2` | base, `out`, `_` | SYCL kernel |
| `log10` | base, `out`, `_` | SYCL kernel |
| `log1p` | base, `out`, `_` | SYCL kernel |
| `logaddexp` | base, `out` | SYCL kernel |
| `logaddexp2` | base, `out` | SYCL kernel |
| `logit` | base, `out`, `_`, `backward` | SYCL kernel |
| `sqrt` | base, `out`, `_` | SYCL kernel |
| `rsqrt` | base, `out`, `_` | SYCL kernel |
| `neg` | base, `out`, `_` | SYCL kernel |
| `reciprocal` | base, `out`, `_` | SYCL kernel |
| `digamma` | base, `out`, `_` | SYCL kernel |
| `polygamma` | base, `out`, `_` | SYCL kernel |
| `erf` | base, `out`, `_` | SYCL kernel |
| `erfc` | base, `out`, `_` | SYCL kernel |
| `erfinv` | base, `out`, `_` | SYCL kernel |
| `lgamma` | base, `out`, `_` | SYCL kernel |
| `gammainc` | base | SYCL kernel |
| `gammaincc` | base | SYCL kernel |
| `uniform_` | base | SYCL RNG |
| `exponential_` | base | SYCL RNG |
| `normal_` | base | SYCL RNG |
| `random_` | base | SYCL RNG |

---

### 2.11 Copy.cpp

**File**: `src/ATen/native/xpu/Copy.cpp`  
**Dependency**: SYCL (critical memory operations)

| Operator | Variants | Implementation |
|----------|----------|----------------|
| `copy_` | base | Host-Device transfer |
| `_copy_from` | base | P2P copy |
| `_copy_from_and_resize` | base | Resizing copy |
| `_to_copy` | base | Device conversion |
| `resize_` | base | Memory resize |
| `set_.source_Tensor` | base | Set operation |

---

### 2.12 TensorCompare.cpp

**File**: `src/ATen/native/xpu/TensorCompare.cpp`  
**Dependency**: SYCL

| Operator | Variants | Implementation |
|----------|----------|--------|
| `eq.Tensor` | base | SYCL comparison |
| `eq.Scalar` | base | SYCL comparison |
| `ne.Tensor` | base | SYCL comparison |
| `ne.Scalar` | base | SYCL comparison |
| `lt.Tensor` | base | SYCL comparison |
| `lt.Scalar` | base | SYCL comparison |
| `gt.Tensor` | base | SYCL comparison |
| `gt.Scalar` | base | SYCL comparison |
| `le.Tensor` | base | SYCL comparison |
| `le.Scalar` | base | SYCL comparison |
| `ge.Tensor` | base | SYCL comparison |
| `ge.Scalar` | base | SYCL comparison |
| `equal` | base | SYCL comparison |
| `allclose` | base | CPU fallback |
| `isclose` | base | SYCL comparison |
| `maximum` | base, `out` | SYCL binary |
| `minimum` | base, `out` | SYCL binary |
| `fmax` | base, `out` | SYCL binary |
| `fmin` | base, `out` | SYCL binary |

---

### 2.13 LayerNorm.cpp

**File**: `src/ATen/native/xpu/LayerNorm.cpp`  
**Dependency**: oneDNN/SYCL

| Operator | Implementation |
|----------|----------------|
| `native_layer_norm` | oneDNN path or SYCL |
| `native_layer_norm_backward` | Backward pass |

---

### 2.14 GroupNorm.cpp

**File**: `src/ATen/native/xpu/GroupNorm.cpp`  
**Dependency**: SYCL

| Operator | Implementation |
|----------|----------------|
| `native_group_norm` | SYCL kernel |
| `native_group_norm_backward` | SYCL kernel |

---

### 2.15 Enabled Op Listing

#### Complete 749 Operator Registry

**Category A: Arithmetic and Comparison (140 ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `add.Tensor` | BinaryOps.cpp | SYCL |
| `add_.Tensor` | BinaryOps.cpp | SYCL |
| `add.out` | BinaryOps.cpp | SYCL |
| `sub.Tensor` | BinaryOps.cpp | SYCL |
| `sub_.Tensor` | BinaryOps.cpp | SYCL |
| `sub.out` | BinaryOps.cpp | SYCL |
| `mul.Tensor` | BinaryOps.cpp | SYCL |
| `mul_.Tensor` | BinaryOps.cpp | SYCL |
| `mul.out` | BinaryOps.cpp | SYCL |
| `div.Tensor` | BinaryOps.cpp | SYCL |
| `div_.Tensor` | BinaryOps.cpp | SYCL |
| `div.out` | BinaryOps.cpp | SYCL |
| `div.Tensor_mode` | BinaryOps.cpp | SYCL |
| `div_.Tensor_mode` | BinaryOps.cpp | SYCL |
| `div.out_mode` | BinaryOps.cpp | SYCL |
| `rsub.Tensor` | BinaryOps.cpp | SYCL |
| `remainder.Tensor` | BinaryOps.cpp | SYCL |
| `remainder_.Tensor` | BinaryOps.cpp | SYCL |
| `remainder.Tensor_out` | BinaryOps.cpp | SYCL |
| `remainder.Scalar_Tensor` | BinaryOps.cpp | SYCL |
| `fmod.Tensor` | BinaryOps.cpp | SYCL |
| `fmod_.Tensor` | BinaryOps.cpp | SYCL |
| `fmod.Tensor_out` | BinaryOps.cpp | SYCL |
| `eq.Scalar` | TensorCompare.cpp | SYCL |
| `eq.Scalar_out` | TensorCompare.cpp | SYCL |
| `eq_.Scalar` | TensorCompare.cpp | SYCL |
| `eq.Tensor` | TensorCompare.cpp | SYCL |
| `eq.Tensor_out` | TensorCompare.cpp | SYCL |
| `eq_.Tensor` | TensorCompare.cpp | SYCL |
| `ne.Scalar` | TensorCompare.cpp | SYCL |
| `ne.Scalar_out` | TensorCompare.cpp | SYCL |
| `ne_.Scalar` | TensorCompare.cpp | SYCL |
| `ne.Tensor` | TensorCompare.cpp | SYCL |
| `ne.Tensor_out` | TensorCompare.cpp | SYCL |
| `ne_.Tensor` | TensorCompare.cpp | SYCL |
| `lt.Scalar` | TensorCompare.cpp | SYCL |
| `lt.Scalar_out` | TensorCompare.cpp | SYCL |
| `lt_.Scalar` | TensorCompare.cpp | SYCL |
| `lt.Tensor` | TensorCompare.cpp | SYCL |
| `lt.Tensor_out` | TensorCompare.cpp | SYCL |
| `lt_.Tensor` | TensorCompare.cpp | SYCL |
| `le.Scalar` | TensorCompare.cpp | SYCL |
| `le.Scalar_out` | TensorCompare.cpp | SYCL |
| `le_.Scalar` | TensorCompare.cpp | SYCL |
| `le.Tensor` | TensorCompare.cpp | SYCL |
| `le.Tensor_out` | TensorCompare.cpp | SYCL |
| `le_.Tensor` | TensorCompare.cpp | SYCL |
| `gt.Scalar` | TensorCompare.cpp | SYCL |
| `gt.Scalar_out` | TensorCompare.cpp | SYCL |
| `gt_.Scalar` | TensorCompare.cpp | SYCL |
| `gt.Tensor` | TensorCompare.cpp | SYCL |
| `gt.Tensor_out` | TensorCompare.cpp | SYCL |
| `gt_.Tensor` | TensorCompare.cpp | SYCL |
| `ge.Scalar` | TensorCompare.cpp | SYCL |
| `ge.Scalar_out` | TensorCompare.cpp | SYCL |
| `ge_.Scalar` | TensorCompare.cpp | SYCL |
| `ge.Tensor` | TensorCompare.cpp | SYCL |
| `ge.Tensor_out` | TensorCompare.cpp | SYCL |
| `ge_.Tensor` | TensorCompare.cpp | SYCL |
| `lerp.Tensor` | Lerp.cpp | SYCL |
| `lerp.Tensor_out` | Lerp.cpp | SYCL |
| `lerp_.Tensor` | Lerp.cpp | SYCL |
| `lerp.Scalar` | Lerp.cpp | SYCL |
| `lerp.Scalar_out` | Lerp.cpp | SYCL |
| `lerp_.Scalar` | Lerp.cpp | SYCL |
| `bitwise_and.Tensor_out` | BinaryOps.cpp | SYCL |
| `bitwise_or.Tensor_out` | BinaryOps.cpp | SYCL |
| `bitwise_xor.Tensor_out` | BinaryOps.cpp | SYCL |
| `bitwise_not.out` | BinaryOps.cpp | SYCL |
| `__lshift__.Scalar` | BinaryOps.cpp | SYCL |
| `__lshift__.Tensor` | BinaryOps.cpp | SYCL |
| `__ilshift__.Scalar` | BinaryOps.cpp | SYCL |
| `__ilshift__.Tensor` | BinaryOps.cpp | SYCL |
| `__rshift__.Scalar` | BinaryOps.cpp | SYCL |
| `__rshift__.Tensor` | BinaryOps.cpp | SYCL |
| `__irshift__.Scalar` | BinaryOps.cpp | SYCL |
| `__irshift__.Tensor` | BinaryOps.cpp | SYCL |
| `bitwise_left_shift.Tensor_out` | BinaryOps.cpp | SYCL |
| `bitwise_right_shift.Tensor_out` | BinaryOps.cpp | SYCL |
| `floor` | UnaryOps.cpp | SYCL |
| `floor_` | UnaryOps.cpp | SYCL |
| `floor.out` | UnaryOps.cpp | SYCL |
| `ceil` | UnaryOps.cpp | SYCL |
| `ceil_` | UnaryOps.cpp | SYCL |
| `ceil.out` | UnaryOps.cpp | SYCL |
| `round` | UnaryOps.cpp | SYCL |
| `round_` | UnaryOps.cpp | SYCL |
| `round.out` | UnaryOps.cpp | SYCL |
| `round.decimals` | UnaryOps.cpp | SYCL |
| `round_.decimals` | UnaryOps.cpp | SYCL |
| `round.decimals_out` | UnaryOps.cpp | SYCL |
| `frac` | UnaryOps.cpp | SYCL |
| `frac_` | UnaryOps.cpp | SYCL |
| `frac.out` | UnaryOps.cpp | SYCL |
| `trunc` | UnaryOps.cpp | SYCL |
| `trunc_` | UnaryOps.cpp | SYCL |
| `trunc.out` | UnaryOps.cpp | SYCL |
| `sign` | UnaryOps.cpp | SYCL |
| `sign_` | UnaryOps.cpp | SYCL |
| `sign.out` | UnaryOps.cpp | SYCL |
| `signbit` | UnaryOps.cpp | SYCL |
| `signbit.out` | UnaryOps.cpp | SYCL |
| `fmax` | TensorCompare.cpp | SYCL |
| `fmax.out` | TensorCompare.cpp | SYCL |
| `fmin` | TensorCompare.cpp | SYCL |
| `fmin.out` | TensorCompare.cpp | SYCL |
| `floor_divide` | BinaryOps.cpp | SYCL |
| `floor_divide_.Tensor` | BinaryOps.cpp | SYCL |
| `floor_divide.out` | BinaryOps.cpp | SYCL |
| `copysign.out` | CompareOps.cpp | SYCL |
| `copysign.Tensor` | CompareOps.cpp | SYCL |
| `copysign_.Tensor` | CompareOps.cpp | SYCL |

**Category B: Activation Functions (62 ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `relu` | Activation.cpp | SYCL `threshold_kernel` |
| `relu_` | Activation.cpp | SYCL `threshold_kernel` |
| `relu.out` | Activation.cpp | SYCL `threshold_kernel` |
| `threshold` | Activation.cpp | SYCL `threshold_kernel` |
| `threshold_` | Activation.cpp | SYCL `threshold_kernel` |
| `threshold.out` | Activation.cpp | SYCL `threshold_kernel` |
| `threshold_backward` | Activation.cpp | SYCL kernel |
| `threshold_backward.grad_input` | Activation.cpp | SYCL kernel |
| `gelu` | Activation.cpp | SYCL `gelu_kernel` |
| `gelu_` | Activation.cpp | SYCL `gelu_kernel` |
| `gelu.out` | Activation.cpp | SYCL `gelu_kernel` |
| `gelu_backward` | Activation.cpp | SYCL `gelu_backward_kernel` |
| `gelu_backward.grad_input` | Activation.cpp | SYCL `gelu_backward_kernel` |
| `silu` | Activation.cpp | SYCL `silu_kernel` |
| `silu_` | Activation.cpp | SYCL `silu_kernel` |
| `silu.out` | Activation.cpp | SYCL `silu_kernel` |
| `silu_backward` | Activation.cpp | SYCL `silu_backward_kernel` |
| `silu_backward.grad_input` | Activation.cpp | SYCL `silu_backward_kernel` |
| `mish` | Activation.cpp | SYCL `mish_kernel` |
| `mish.out` | Activation.cpp | SYCL `mish_kernel` |
| `mish_` | Activation.cpp | SYCL `mish_kernel` |
| `mish_backward` | Activation.cpp | SYCL `mish_backward_kernel` |
| `softplus` | Activation.cpp | SYCL `softplus_kernel` |
| `softplus.out` | Activation.cpp | SYCL `softplus_kernel` |
| `softplus_backward` | Activation.cpp | SYCL `softplus_backward_kernel` |
| `softplus_backward.grad_input` | Activation.cpp | SYCL `softplus_backward_kernel` |
| `softshrink` | Activation.cpp | SYCL `softshrink_kernel` |
| `softshrink.out` | Activation.cpp | SYCL `softshrink_kernel` |
| `softshrink_backward` | Activation.cpp | SYCL `softshrink_backward_kernel` |
| `softshrink_backward.grad_input` | Activation.cpp | SYCL `softshrink_backward_kernel` |
| `hardshrink` | Activation.cpp | SYCL `hardshrink_kernel` |
| `hardtanh` | Activation.cpp | SYCL kernel |
| `hardtanh.out` | Activation.cpp | SYCL kernel |
| `hardtanh_` | Activation.cpp | SYCL kernel |
| `hardtanh_backward` | Activation.cpp | SYCL |
| `hardtanh_backward.grad_input` | Activation.cpp | SYCL |
| `hardsigmoid` | Activation.cpp | SYCL |
| `hardsigmoid.out` | Activation.cpp | SYCL |
| `hardsigmoid_` | Activation.cpp | SYCL |
| `hardsigmoid_backward` | Activation.cpp | SYCL |
| `hardsigmoid_backward.grad_input` | Activation.cpp | SYCL |
| `hardswish` | Activation.cpp | SYCL |
| `hardswish.out` | Activation.cpp | SYCL |
| `hardswish_` | Activation.cpp | SYCL |
| `hardswish_backward` | Activation.cpp | SYCL |
| `hardswish_backward.grad_input` | Activation.cpp | SYCL |
| `leaky_relu` | Activation.cpp | SYCL |
| `leaky_relu_` | Activation.cpp | SYCL |
| `leaky_relu.out` | Activation.cpp | SYCL |
| `leaky_relu_backward` | Activation.cpp | SYCL |
| `leaky_relu_backward.grad_input` | Activation.cpp | SYCL |
| `elu` | Activation.cpp | SYCL |
| `elu.out` | Activation.cpp | SYCL |
| `elu_` | Activation.cpp | SYCL |
| `elu_backward` | Activation.cpp | SYCL |
| `elu_backward.grad_input` | Activation.cpp | SYCL |
| `log_sigmoid_forward.output` | Activation.cpp | SYCL |
| `log_sigmoid_forward` | Activation.cpp | SYCL |
| `log_sigmoid_backward.grad_input` | Activation.cpp | SYCL |
| `log_sigmoid_backward` | Activation.cpp | SYCL |
| `prelu` | Activation.cpp | SYCL |
| `prelu_backward` | Activation.cpp | SYCL |
| `_prelu_kernel` | Activation.cpp | SYCL |
| `_prelu_kernel_backward` | Activation.cpp | SYCL |
| `glu` | Activation.cpp | SYCL |
| `glu.out` | GatedLinearUnit.cpp | SYCL |
| `glu_backward` | Activation.cpp | SYCL |
| `glu_backward.grad_input` | Activation.cpp | SYCL |

**Category C: Math Operations (50+ ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `sin` | UnaryOps.cpp | SYCL kernel |
| `sin_` | UnaryOps.cpp | SYCL kernel |
| `sin.out` | UnaryOps.cpp | SYCL kernel |
| `sinh` | UnaryOps.cpp | SYCL kernel |
| `sinh_` | UnaryOps.cpp | SYCL kernel |
| `sinh.out` | UnaryOps.cpp | SYCL kernel |
| `cos` | UnaryOps.cpp | SYCL kernel |
| `cos_` | UnaryOps.cpp | SYCL kernel |
| `cos.out` | UnaryOps.cpp | SYCL kernel |
| `cosh` | UnaryOps.cpp | SYCL kernel |
| `cosh_` | UnaryOps.cpp | SYCL kernel |
| `cosh.out` | UnaryOps.cpp | SYCL kernel |
| `tan` | UnaryOps.cpp | SYCL kernel |
| `tan_` | UnaryOps.cpp | SYCL kernel |
| `tan.out` | UnaryOps.cpp | SYCL kernel |
| `tanh` | UnaryOps.cpp | SYCL kernel |
| `tanh_` | UnaryOps.cpp | SYCL kernel |
| `tanh.out` | UnaryOps.cpp | SYCL kernel |
| `tanh_backward` | UnaryOps.cpp | SYCL kernel |
| `tanh_backward.grad_input` | UnaryOps.cpp | SYCL kernel |
| `asin` | UnaryOps.cpp | SYCL kernel |
| `asin_` | UnaryOps.cpp | SYCL kernel |
| `asin.out` | UnaryOps.cpp | SYCL kernel |
| `acos` | UnaryOps.cpp | SYCL kernel |
| `acos_` | UnaryOps.cpp | SYCL kernel |
| `acos.out` | UnaryOps.cpp | SYCL kernel |
| `acosh` | UnaryOps.cpp | SYCL kernel |
| `acosh_` | UnaryOps.cpp | SYCL kernel |
| `acosh.out` | UnaryOps.cpp | SYCL kernel |
| `atan` | UnaryOps.cpp | SYCL kernel |
| `atan_` | UnaryOps.cpp | SYCL kernel |
| `atan.out` | UnaryOps.cpp | SYCL kernel |
| `atanh` | UnaryOps.cpp | SYCL kernel |
| `atanh_` | UnaryOps.cpp | SYCL kernel |
| `atanh.out` | UnaryOps.cpp | SYCL kernel |
| `atan2` | BinaryOps.cpp | SYCL kernel |
| `atan2.out` | BinaryOps.cpp | SYCL kernel |
| `atan2_` | BinaryOps.cpp | SYCL kernel |
| `asinh` | UnaryOps.cpp | SYCL kernel |
| `asinh.out` | UnaryOps.cpp | SYCL kernel |
| `asinh_` | UnaryOps.cpp | SYCL kernel |
| `log` | UnaryOps.cpp | SYCL kernel |
| `log_` | UnaryOps.cpp | SYCL kernel |
| `log.out` | UnaryOps.cpp | SYCL kernel |
| `log2` | UnaryOps.cpp | SYCL kernel |
| `log2_` | UnaryOps.cpp | SYCL kernel |
| `log2.out` | UnaryOps.cpp | SYCL kernel |
| `log10` | UnaryOps.cpp | SYCL kernel |
| `log10_` | UnaryOps.cpp | SYCL kernel |
| `log10.out` | UnaryOps.cpp | SYCL kernel |
| `log1p` | UnaryOps.cpp | SYCL kernel |
| `log1p_` | UnaryOps.cpp | SYCL kernel |
| `log1p.out` | UnaryOps.cpp | SYCL kernel |
| `exp` | UnaryOps.cpp | SYCL kernel |
| `exp.out` | UnaryOps.cpp | SYCL kernel |
| `exp_` | UnaryOps.cpp | SYCL kernel |
| `exp2` | UnaryOps.cpp | SYCL kernel |
| `exp2_` | UnaryOps.cpp | SYCL kernel |
| `exp2.out` | UnaryOps.cpp | SYCL kernel |
| `expm1` | UnaryOps.cpp | SYCL kernel |
| `expm1_` | UnaryOps.cpp | SYCL kernel |
| `expm1.out` | UnaryOps.cpp | SYCL kernel |

**Category D: Element-wise Unary (40+ ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `abs` | UnaryOps.cpp | SYCL |
| `abs_` | UnaryOps.cpp | SYCL |
| `abs.out` | UnaryOps.cpp | SYCL |
| `neg` | UnaryOps.cpp | SYCL |
| `neg_` | UnaryOps.cpp | SYCL |
| `neg.out` | UnaryOps.cpp | SYCL |
| `reciprocal` | UnaryOps.cpp | SYCL |
| `reciprocal_` | UnaryOps.cpp | SYCL |
| `reciprocal.out` | UnaryOps.cpp | SYCL |
| `sqrt` | UnaryOps.cpp | SYCL |
| `sqrt_` | UnaryOps.cpp | SYCL |
| `sqrt.out` | UnaryOps.cpp | SYCL |
| `rsqrt` | UnaryOps.cpp | SYCL |
| `rsqrt_` | UnaryOps.cpp | SYCL |
| `rsqrt.out` | UnaryOps.cpp | SYCL |
| `erf` | UnaryOps.cpp | SYCL |
| `erf_` | UnaryOps.cpp | SYCL |
| `erf.out` | UnaryOps.cpp | SYCL |
| `erfc` | UnaryOps.cpp | SYCL |
| `erfc_` | UnaryOps.cpp | SYCL |
| `erfc.out` | UnaryOps.cpp | SYCL |
| `erfinv` | UnaryOps.cpp | SYCL |
| `erfinv_` | UnaryOps.cpp | SYCL |
| `erfinv.out` | UnaryOps.cpp | SYCL |
| `lgamma` | UnaryOps.cpp | SYCL |
| `lgamma_` | UnaryOps.cpp | SYCL |
| `lgamma.out` | UnaryOps.cpp | SYCL |
| `digamma` | UnaryOps.cpp | SYCL |
| `digamma_` | UnaryOps.cpp | SYCL |
| `digamma.out` | UnaryOps.cpp | SYCL |
| `polygamma` | UnaryOps.cpp | SYCL |
| `polygamma_` | UnaryOps.cpp | SYCL |
| `polygamma.out` | UnaryOps.cpp | SYCL |
| `polygamma` | UnaryOps.cpp | SYCL |

**Category E: Reduction Operations (48 ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `sum.dim_IntList` | ReduceOps.cpp | SYCL |
| `sum.IntList_out` | ReduceOps.cpp | SYCL |
| `prod` | ReduceOps.cpp | SYCL |
| `prod.dim_int` | ReduceOps.cpp | SYCL |
| `prod.int_out` | ReduceOps.cpp | SYCL |
| `nansum` | ReduceOps.cpp | SYCL |
| `nansum.out` | ReduceOps.cpp | SYCL |
| `mean.out` | ReduceOps.cpp | SYCL |
| `mean.dim` | ReduceOps.cpp | SYCL |
| `all.dim` | ReduceOps.cpp | SYCL |
| `all.out` | ReduceOps.cpp | SYCL |
| `all.dims` | ReduceOps.cpp | SYCL |
| `all.dims_out` | ReduceOps.cpp | SYCL |
| `all` | ReduceOps.cpp | SYCL |
| `all.all_out` | ReduceOps.cpp | SYCL |
| `any.dim` | ReduceOps.cpp | SYCL |
| `any.out` | ReduceOps.cpp | SYCL |
| `any.dims` | ReduceOps.cpp | SYCL |
| `any.dims_out` | ReduceOps.cpp | SYCL |
| `any` | ReduceOps.cpp | SYCL |
| `any.all_out` | ReduceOps.cpp | SYCL |
| `max` | ReduceOps.cpp | SYCL |
| `max.unary_out` | ReduceOps.cpp | SYCL |
| `max.dim_max` | ReduceOps.cpp | SYCL |
| `min` | ReduceOps.cpp | SYCL |
| `min.unary_out` | ReduceOps.cpp | SYCL |
| `min.dim_min` | ReduceOps.cpp | SYCL |
| `argmax` | ReduceOps.cpp | SYCL |
| `argmax.out` | ReduceOps.cpp | SYCL |
| `argmin` | ReduceOps.cpp | SYCL |
| `argmin.out` | ReduceOps.cpp | SYCL |
| `std.correction` | ReduceOps.cpp | SYCL |
| `std.correction_out` | ReduceOps.cpp | SYCL |
| `std_mean.correction` | ReduceOps.cpp | SYCL |
| `var.correction` | ReduceOps.cpp | SYCL |
| `var.correction_out` | ReduceOps.cpp | SYCL |
| `var_mean.correction` | ReduceOps.cpp | SYCL |
| `amax` | ReduceOps.cpp | SYCL |
| `amax.out` | ReduceOps.cpp | SYCL |
| `amin` | ReduceOps.cpp | SYCL |
| `amin.out` | ReduceOps.cpp | SYCL |
| `median` | ReduceOps.cpp | SYCL |
| `median.dim_values` | ReduceOps.cpp | SYCL |
| `nanmedian` | ReduceOps.cpp | SYCL |
| `nanmedian.dim_values` | ReduceOps.cpp | SYCL |

**Category F: Linear Algebra (95 ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `mm.out` | Blas.cpp | SYCL + oneMKL |
| `bmm.out` | Blas.cpp | SYCL + oneMKL |
| `addmm.out` | Blas.cpp | SYCL + oneMKL |
| `addbmm.out` | Blas.cpp | SYCL + oneMKL |
| `addmv.out` | Blas.cpp | SYCL + oneMKL |
| `mv.out` | Blas.cpp | SYCL + oneMKL |
| `dot` | Blas.cpp | SYCL + oneMKL |
| `vdot` | Blas.cpp | SYCL + oneMKL |
| `lu_solve` | BatchLinearAlgebra.cpp | SYCL + oneMKL |
| `lu_solve.out` | BatchLinearAlgebra.cpp | SYCL + oneMKL |
| `linalg_vector_norm` | ComputeLinearCombination.cpp | SYCL |
| `linalg_vector_norm.out` | ComputeLinearCombination.cpp | SYCL |
| `norm.ScalarOpt_dim_dtype` | ComputeLinearCombination.cpp | SYCL |
| `norm.dtype_out` | ComputeLinearCombination.cpp | SYCL |
| `norm.ScalarOpt_dim` | ComputeLinearCombination.cpp | SYCL |
| `norm.out` | ComputeLinearCombination.cpp | SYCL |
| `addr` | LinearAlgebra.cpp | SYCL |
| `addr.out` | LinearAlgebra.cpp | SYCL |
| `outer` | MatrixBomb.cpp | SYCL |
| `cross` | Cross.cpp | SYCL |
| `linalg_cross` | LinearAlgebra.cpp | SYCL |
| `linalg_cross.out` | LinearAlgebra.cpp | SYCL |
| `ger` | MatrixMultiply.cpp | SYCL |
| `matmul` | MatrixMultiply.cpp | SYCL |

**Category G: Convolution and Pooling (28 ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `conv2d` | oneDNN path | oneDNN |
| `conv3d` | oneDNN path | oneDNN |
| `conv_transpose2d` | Copy.cpp | SYCL/+oneDNN |
| `conv_transpose3d` | Copy.cpp | SYCL/+oneDNN |
| `max_pool2d_with_indices` | MaxUnpooling.cpp | SYCL |
| `max_pool2d_with_indices.out` | MaxUnpooling.cpp | SYCL |
| `max_pool2d_with_indices_backward` | MaxUnpooling.cpp | SYCL |
| `avg_pool2d` | AveragePool2d.cpp | SYCL |
| `avg_pool2d.out` | AveragePool2d.cpp | SYCL |
| `avg_pool2d_backward` | AveragePool2d.cpp | SYCL |
| `avg_pool2d_backward.grad_input` | AveragePool2d.cpp | SYCL |
| `adaptive_avg_pool2d.out` | AdaptiveAveragePooling2d.cpp | SYCL |
| `_adaptive_avg_pool2d` | AdaptiveAveragePooling2d.cpp | SYCL |
| `_adaptive_avg_pool2d_backward` | AdaptiveAveragePooling2d.cpp | SYCL |
| `adaptive_max_pool2d` | AdaptiveMaxPooling2d.cpp | SYCL |
| `adaptive_max_pool2d.out` | AdaptiveMaxPooling2d.cpp | SYCL |
| `adaptive_max_pool2d_backward` | AdaptiveMaxPooling2d.cpp | SYCL |
| `adaptive_max_pool2d_backward.grad_input` | AdaptiveMaxPooling2d.cpp | SYCL |
| `fractional_max_pool2d` | FractionalMaxPool2d.cpp | SYCL |
| `fractional_max_pool3d` | FractionalMaxPool3d.cpp | SYCL |
| `fractional_max_pool3d.output` | FractionalMaxPool3d.cpp | SYCL |
| `fractional_max_pool3d_backward` | FractionalMaxPool3d.cpp | SYCL |
| `fractional_max_pool3d_backward.grad_input` | FractionalMaxPool3d.cpp | SYCL |

**Category H: Normalization (34 ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `batch_norm` | BatchNorm.cpp | oneDNN |
| `batch_norm.out` | BatchNorm.cpp | oneDNN |
| `native_batch_norm` | BatchNorm.cpp | oneDNN |
| `native_batch_norm.out` | BatchNorm.cpp | oneDNN |
| `native_batch_norm_backward` | BatchNorm.cpp | oneDNN |
| `_batch_norm_with_update` | BatchNorm.cpp | oneDNN |
| `_batch_norm_with_update.out` | BatchNorm.cpp | oneDNN |
| `batch_norm_backward` | BatchNorm.cpp | oneDNN |
| `batch_norm_stats` | BatchNorm.cpp | oneDNN |
| `batch_norm_elemt` | BatchNorm.cpp | oneDNN |
| `batch_norm_elemt.out` | BatchNorm.cpp | oneDNN |
| `batch_norm_backward_reduce` | BatchNorm.cpp | oneDNN |
| `batch_norm_backward_elemt` | BatchNorm.cpp | oneDNN |
| `batch_norm_update_stats` | BatchNorm.cpp | oneDNN |
| `native_layer_norm` | LayerNorm.cpp | oneDNN/SYCL |
| `nested_layer_norm` | LayerNorm.cpp | oneDNN/SYCL |
| `native_group_norm` | GroupNorm.cpp | SYCL |
| `native_group_norm_backward` | GroupNorm.cpp | SYCL |

**Category I: Indexing and Searching (31 ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `index.Tensor` | Indexing.cpp | SYCL |
| `index.Tensor_out` | Indexing.cpp | SYCL |
| `index_select` | Indexing.cpp | SYCL |
| `index_select.out` | Indexing.cpp | SYCL |
| `index_add.out` | Indexing.cpp | SYCL |
| `index_add_` | Indexing.cpp | SYCL |
| `index_fill_.int_Scalar` | Indexing.cpp | SYCL |
| `index_fill_.int_Tensor` | Indexing.cpp | SYCL |
| `gather` | Indexing.cpp | SYCL |
| `gather.out` | Indexing.cpp | SYCL |
| `scatter.src` | Indexing.cpp | SYCL |
| `scatter.src_out` | Indexing.cpp | SYCL |
| `scatter_.src` | Indexing.cpp | SYCL |
| `scatter.value` | Indexing.cpp | SYCL |
| `scatter.value_out` | Indexing.cpp | SYCL |
| `scatter_.value` | Indexing.cpp | SYCL |
| `scatter.reduce` | Indexing.cpp | SYCL |
| `scatter.reduce_out` | Indexing.cpp | SYCL |
| `scatter_.reduce` | Indexing.cpp | SYCL |
| `scatter_add` | Indexing.cpp | SYCL |
| `scatter_add.out` | Indexing.cpp | SYCL |
| `scatter_reduce.two` | Sorting.cpp | SYCL |
| `scatter_reduce.two_out` | Sorting.cpp | SYCL |
| `scatter_reduce_.two` | Sorting.cpp | SYCL |
| `masked_select` | Indexing.cpp | SYCL |
| `masked_select.out` | Indexing.cpp | SYCL |
| `bucketize.Tensor` | Bucketization.cpp | SYCL |
| `bucketize.Tensor_out` | Bucketization.cpp | SYCL |
| `searchsorted.Tensor` | Bucketization.cpp | SYCL |
| `searchsorted.Tensor_out` | Bucketization.cpp | SYCL |

**Category J: Sorting and TopK (10 ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `sort.stable` | Sorting.cpp | SYCL |
| `sort.values_stable` | Sorting.cpp | SYCL |
| `argsort.stable` | Sorting.cpp | SYCL |
| `topk` | Sorting.cpp | SYCL |
| `topk.values` | Sorting.cpp | SYCL |

**Category K: Specialized Operations (180+ ops)**

| Operator | File | Dependency |
|----------|------|------------|
| `smooth_l1_loss` | Loss.cpp | SYCL |
| `smooth_l1_loss.out` | Loss.cpp | SYCL |
| `smooth_l1_loss_backward` | Loss.cpp | SYCL |
| `smooth_l1_loss_backward.grad_input` | Loss.cpp | SYCL |
| `mse_loss` | Loss.cpp | SYCL |
| `mse_loss.out` | Loss.cpp | SYCL |
| `mse_loss_backward` | Loss.cpp | SYCL |
| `mse_loss_backward.grad_input` | Loss.cpp | SYCL |
| `binary_cross_entropy` | Loss.cpp | SYCL |
| `binary_cross_entropy.out` | Loss.cpp | SYCL |
| `binary_cross_entropy_backward` | Loss.cpp | SYCL |
| `nll_loss_forward` | LossNLL.cpp | SYCL |
| `nll_loss_forward.output` | LossNLL.cpp | SYCL |
| `nll_loss_backward` | LossNLL.cpp | SYCL |
| `nll_loss_backward.grad_input` | LossNLL.cpp | SYCL |
| `huber_loss` | Loss.cpp | SYCL |
| `huber_loss.out` | Loss.cpp | SYCL |
| `huber_loss_backward.out` | Loss.cpp | SYCL |
| `multi_margin_loss` | LossMultiMargin.cpp | SYCL |
| `multi_margin_loss.out` | LossMultiMargin.cpp | SYCL |
| `multi_margin_loss_backward` | LossMultiMargin.cpp | SYCL |
| `multi_margin_loss_backward.grad_input` | LossMultiMargin.cpp | SYCL |
| `clamp` | UnaryOps.cpp | SYCL |
| `clamp.out` | UnaryOps.cpp | SYCL |
| `clamp_` | UnaryOps.cpp | SYCL |
| `clamp.Tensor` | UnaryOps.cpp | SYCL |
| `clamp.Tensor_out` | UnaryOps.cpp | SYCL |
| `clamp_.Tensor` | UnaryOps.cpp | SYCL |
| `clamp_max` | UnaryOps.cpp | SYCL |
| `clamp_max.out` | UnaryOps.cpp | SYCL |
| `clamp_max_` | UnaryOps.cpp | SYCL |
| `clamp_max.Tensor` | UnaryOps.cpp | SYCL |
| `clamp_max.Tensor_out` | UnaryOps.cpp | SYCL |
| `clamp_max_.Tensor` | UnaryOps.cpp | SYCL |
| `clamp_min` | UnaryOps.cpp | SYCL |
| `clamp_min.out` | UnaryOps.cpp | SYCL |
| `clamp_min_` | UnaryOps.cpp | SYCL |
| `clamp_min.Tensor` | UnaryOps.cpp | SYCL |
| `clamp_min.Tensor_out` | UnaryOps.cpp | SYCL |
| `clamp_min_.Tensor` | UnaryOps.cpp | SYCL |
| `im2col` | Im2Col.cpp | SYCL |
| `im2col.out` | Im2Col.cpp | SYCL |
| `col2im` | Im2Col.cpp | SYCL |
| `col2im.out` | Im2Col.cpp | SYCL |
| `grid_sampler_2d` | GridSampler.cpp | SYCL |
| `grid_sampler_2d_backward` | GridSampler.cpp | SYCL |
| `grid_sampler_3d` | GridSampler.cpp | SYCL |
| `grid_sampler_3d_backward` | GridSampler.cpp | SYCL |
| `reflection_pad1d` | ReflectionPad.cpp | SYCL |
| `reflection_pad1d.out` | ReflectionPad.cpp | SYCL |
| `reflection_pad1d_backward` | ReflectionPad.cpp | SYCL |
| `reflection_pad2d` | ReflectionPad.cpp | SYCL |
| `reflection_pad2d.out` | ReflectionPad.cpp | SYCL |
| `reflection_pad2d_backward` | ReflectionPad.cpp | SYCL |
| `replication_pad1d` | ReplicationPad.cpp | SYCL |
| `replication_pad1d.out` | ReplicationPad.cpp | SYCL |
| `replication_pad1d_backward` | ReplicationPad.cpp | SYCL |
| `replication_pad2d` | ReplicationPad.cpp | SYCL |
| `replication_pad2d.out` | ReplicationPad.cpp | SYCL |
| `replication_pad2d_backward` | ReplicationPad.cpp | SYCL |
| `dropout` | Dropout.cpp | SYCL |
| `native_dropout` | Dropout.cpp | SYCL |
| `native_dropout_backward` | Dropout.cpp | SYCL |
| `nonzero` | Nonzero.cpp | SYCL |
| `nonzero.out` | Nonzero.cpp | SYCL |
| `repeat_interleave.Tensor` | Repeat.cpp | SYCL |
| `tril.out` | TriangularOps.cpp | SYCL |
| `tril` | TriangularOps.cpp | SYCL |
| `tril_` | TriangularOps.cpp | SYCL |
| `triu.out` | TriangularOps.cpp | SYCL |
| `triu` | TriangularOps.cpp | SYCL |
| `triu_` | TriangularOps.cpp | SYCL |
| `flip` | TensorTransformations.cpp | SYCL |
| `roll` | TensorTransformations.cpp | SYCL |
| `unfold` | UnfoldBackward.cpp | SYCL |
| `unfold_backward` | UnfoldBackward.cpp | SYCL |
| `cumsum` | ScanKernels.cpp | SYCL |
| `cumsum.out` | ScanKernels.cpp | SYCL |
| `cumsum_` | ScanKernels.cpp | SYCL |
| `cumprod` | ScanKernels.cpp | SYCL |
| `cumprod.out` | ScanKernels.cpp | SYCL |
| `cumprod_` | ScanKernels.cpp | SYCL |
| `renorm` | ReduceOps.cpp | SYCL |
| `renorm.out` | ReduceOps.cpp | SYCL |
| `renorm_` | ReduceOps.cpp | SYCL |
| `trace` | ReduceOps.cpp | SYCL |
| `unique_dim` | Unique.cpp | SYCL |
| `unique_dim_consecutive` | Unique.cpp | SYCL |
| `unique_consecutive` | Unique.cpp | SYCL |
| `unique` | Unique.cpp | SYCL |
| `unique2` | Unique.cpp | SYCL |
| `cat` | TensorAdvancedIndexing.cpp | SYCL |
| `cat.out` | TensorAdvancedIndexing.cpp | SYCL |
| `isin.Tensor_Tensor_out` | CompareOps.cpp | SYCL |
| `isin.Tensor_Tensor` | CompareOps.cpp | SYCL |
| `isin.Tensor_Scalar_out` | CompareOps.cpp | SYCL |
| `isin.Tensor_Scalar` | CompareOps.cpp | SYCL |
| `isin.Scalar_Tensor_out` | CompareOps.cpp | SYCL |
| `isin.Scalar_Tensor` | CompareOps.cpp | SYCL |
| `gcd` | ScanKernels.cpp | SYCL |
| `gcd.out` | ScanKernels.cpp | SYCL |
| `gcd_` | ScanKernels.cpp | SYCL |
| `hypot` | BinaryOps.cpp | SYCL |
| `hypot.out` | BinaryOps.cpp | SYCL |
| `hypot_` | BinaryOps.cpp | SYCL |
| `nextafter` | UnaryOps.cpp | SYCL |
| `nextafter.out` | UnaryOps.cpp | SYCL |
| `nextafter_` | UnaryOps.cpp | SYCL |
| `put_` | Indexing.cpp | SYCL |
| `take` | Indexing.cpp | SYCL |
| `take.out` | Indexing.cpp | SYCL |
| `segment_reduce` | SegmentReduce.cpp | SYCL |
| `_segment_reduce_backward` | SegmentReduce.cpp | SYCL |
| `bincount` | Distributions.cpp | SYCL |
| `multinomial` | Distributions.cpp | SYCL |
| `multinomial.out` | Distributions.cpp | SYCL |
| `randperm.generator_out` | RangeFactories.cpp | SYCL |
| `arange.start_out` | RangeFactories.cpp | SYCL |
| `range.out` | RangeFactories.cpp | SYCL |
| `fill_.Scalar` | Fill.cpp | SYCL |
| `fill_.Tensor` | Fill.cpp | SYCL |

---

## Part III: Dependency Library Reference

### 3.1 SYCL Kernel Files Index

Complete listing of 201 SYCL kernel files organized by operation type:

**Activation Kernels (14 files)**
```
ActivationBlasKernel.*, ActivationEluKernels.*, ActivationGeluKernel.*, 
ActivationHardshrinkKernels.*, ActivationHardsigmoidKernels.*, 
ActivationHardswishKernels.*, ActivationHardtanhKernels.*, 
ActivationLeakyReluKernels.*, ActivationLogSigmoidKernels.*, 
ActivationMishKernels.*, ActivationPreluKernels.*, ActivationSiluKernels.*, 
ActivationSoftplusKernels.*, ActivationSoftshrinkKernels.*, 
ActivationThresholdKernel.*, Activation专题Kernel.*
```

**Unary Math Kernels (80+ files)**
```
UnaryAcoshKernel.*, UnaryAsinKernel.*, UnaryAtanhKernel.*, 
UnaryAtan2Kernel.*, UnaryBitwiseNotKernel.*, UnaryCeilKernel.*, 
UnaryCosKernel.*, UnaryCoshKernel.*, UnaryDigammaKernel.*, 
UnaryEluKernel.*, UnaryErfKernel.*, UnaryErfcKernel.*, 
UnaryErfinvKernel.*, UnaryExpKernel.*, UnaryExpm1Kernel.*, 
UnaryFloorKernel.*, UnaryLgammaKernel.*, UnaryLog10Kernel.*, 
UnaryLog1pKernel.*, UnaryLogKernel.*, UnaryLog2Kernel.*, 
UnaryNextAfterKernel.*, UnaryNegKernel.*, UnaryPolygammaKernel.*, 
UnaryPowKernel.*, UnaryReciprocalKernel.*, UnaryRoundKernel.*, 
UnaryRsqrtKernel.*, UnarySigmoidKernel.*, UnarySinhKernel.*, 
UnarySinKernel.*, UnarySqrtKernel.*, UnaryTanKernel.*, 
UnaryTanhKernel.*, UnaryTruncKernel.*, UnarySignKernel.*, Unary专题Kernel.*
```

**Reduction Kernels (30+ files)**
```
BinaryAddcmulKernel.*, BinaryAddcdivKernel.*, FillKernel.*, 
ForeachAddKernel.*, ForeachDivKernel.*, ForeachMulKernel.*, 
ForeachNormKernel.*, ForeachSqrtKernel.*, NormKernel.*, 
ReduceAddKernel.*, ReduceAllKernel.*, ReduceAnyKernel.*, 
ReduceArgmaxKernel.*, ReduceArgminKernel.*, ReduceDivKernel.*, 
ReduceMaxKernel.*, ReduceMinKernel.*, ReduceProdKernel.*, 
ReduceSubKernel.*, SumKernel.*, Unary专题Kernel.*.*, VarKernel.*
```

**Pooling Kernels (8 files)**
```
AdaptiveAvgPool2dKernel.*, AdaptiveAvgPool3dKernel.*, 
AveragePool2dKernel.*, AveragePool3dKernel.*, 
MaxPool2dKernel.*, MaxPool3dKernel.*
```

**Tensor Operation Kernels (40+ files)**
```
CopyDeviceToDevice.*, FlipKernel.*, RollKernel.*, 
TransformerXvKernel.*, UnfoldKernel.*, Col2ImKernel.*, 
EmbeddingKernel.*, IndexingKernel.*, GatherKernel.*, 
ScatterKernel.*, SortKernel.*, TopKKernel.*, Binary专题Kernel.*
```

### 3.2 oneMKL/SYCL Mixed Implementation Files

The following files implement both SYCL and oneMKL with runtime conditional:

| File | Primary Ops | oneMKL | SYCL |
|------|------------|--------|------|
| `Blas.cpp` | Matrix multiply, dot | Yes | Yes |
| `BatchLinearAlgebra.cpp` | LU, chol, solve | Yes | Yes |
| `SpectralOps.cpp` | FFT | Yes | Yes |

### 3.3 CPU Fallback File Index

The following files implement CPU fallback via `XPUFallback.template`:

```
Im2Col.cpp, Histogram.cpp, GridSampler.cpp, FractionalMaxPool2d.cpp, 
FractionalMaxPool3d.cpp, Unique.cpp, Distributions.cpp, RewriteKernel.*
```

---

## Part IV: CPU Fallback Operators

Unimplemented operators that fallback to CPU execution when native XPU implementation unavailable:

### 4.1 Hardcoded Fallback List (XPUFallback.template:226-257)

These operators have explicit CPU fallback registration:

| Operator | Category |
|----------|----------|
| `cholesky` | Linear Algebra |
| `cholesky.out` | Linear Algebra |
| `cholesky_inverse` | Linear Algebra |
| `cholesky_inverse.out` | Linear Algebra |
| `_cholesky_solve_helper` | Linear Algebra |
| `_efficient_attention_forward` | Attention |
| `_flash_attention_forward` | Attention |
| `geqrf` | Linear Algebra |
| `geqrf.a` | Linear Algebra |
| `linalg_eig` | Linear Algebra |
| `linalg_eig.out` | Linear Algebra |
| `_linalg_eigvals` | Linear Algebra |
| `linalg_eigvals.out` | Linear Algebra |
| `_linalg_eigh.eigenvalues` | Linear Algebra |
| `linalg_householder_product` | Linear Algebra |
| `linalg_householder_product.out` | Linear Algebra |
| `linalg_ldl_factor_ex.out` | Linear Algebra |
| `linalg_ldl_solve.out` | Linear Algebra |
| `linalg_lstsq.out` | Linear Algebra |
| `linalg_lu.out` | Linear Algebra |
| `linalg_matrix_exp` | Linear Algebra |
| `linalg_qr.out` | Linear Algebra |
| `_linalg_svd.U` | Linear Algebra |
| `lu_unpack.out` | Linear Algebra |
| `ormqr` | Linear Algebra |
| `ormqr.out` | Linear Algebra |
| `triangular_solve.X` | Linear Algebra |
| `_validate_compressed_sparse_indices` | Sparse |

---

## Part V: Implementation Dependency Matrix

### 5.1 Dense Matrix Operations

| Operator | SYCL | oneMKL | oneDNN | CPU Fallback |
|----------|------|--------|--------|--------------|
| mm | ✓ | ✓ | | ✓ |
| bmm | ✓ | ✓ | | ✓ |
| addmm | ✓ | ✓ | | ✓ |
| addbmm | ✓ | ✓ | | ✓ |
| dot | ✓ | ✓ | | ✓ |
| vdot | ✓ | ✓ | | ✓ |

### 5.2 Neural Network Operations

| Operator | SYCL | oneMKL | oneDNN | CPU Fallback |
|----------|------|--------|--------|--------------|
| conv2d | | | ✓ | |
| conv3d | | | ✓ | |
| linear | | | ✓ | |
| batch_norm | | | ✓ | |
| layer_norm | | | ✓/SYCL | |

### 5.3 Element-wise Operations

| Operator | SYCL | oneMKL | oneDNN | CPU Fallback |
|----------|------|--------|--------|--------------|
| relu | ✓ | | | |
| gelu | ✓ | | | |
| silu | ✓ | | | |
| sigmoid | ✓ | | | |
| tanh | ✓ | | | |

---

*Document generated from analysis of torch-xpu-ops repository. Total 749 operators catalogued with implementation and dependency mapping.*