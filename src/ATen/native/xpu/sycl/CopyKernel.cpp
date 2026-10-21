/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <comm/xpu_aten.h>

#include <ATen/Dispatch_v2.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/CopyKernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct CopyScalarFunc {
  scalar_t operator()(scalar_t src_val) const {
    return src_val;
  }
};

template <typename in_scalar_t, typename out_scalar_t>
struct CastScalarFunc {
  out_scalar_t operator()(in_scalar_t src_val) const {
    return (out_scalar_t)src_val;
  }
};

// TODO: Avoid using sycl::half to prevent the fp16->fp32->fp8 fusion
// from incorrectly converting -0.0 to NaN. This temporary fix should
// be removed once the compiler/driver error is resolved.
template <typename Float8DataType>
struct CastScalarFunc<Half, Float8DataType> {
  Float8DataType operator()(Half src_val) const {
    Half val = src_val == Half(-0.0) ? Half(0.0) : src_val;
    return Float8DataType(val);
  }
};

void float8_copy_kernel_xpu(TensorIteratorBase& iter) {
  ScalarType dtype = iter.dtype(0);
  ScalarType other_dtype = iter.dtype(1);
  if (dtype == kFloat8_e4m3fn) {
    switch (other_dtype) {
      case kFloat:
        gpu_kernel_nocast(iter, CastScalarFunc<float, Float8_e4m3fn>());
        break;
      case kHalf:
        gpu_kernel_nocast(iter, CastScalarFunc<Half, Float8_e4m3fn>());
        break;
      case kBFloat16:
        gpu_kernel_nocast(iter, CastScalarFunc<BFloat16, Float8_e4m3fn>());
        break;
      default:
        gpu_kernel(iter, CopyScalarFunc<Float8_e4m3fn>());
        break;
    }
  } else if (dtype == kFloat8_e5m2) {
    switch (other_dtype) {
      case kFloat:
        gpu_kernel_nocast(iter, CastScalarFunc<float, Float8_e5m2>());
        break;
      case kHalf:
        gpu_kernel_nocast(iter, CastScalarFunc<Half, Float8_e5m2>());
        break;
      case kBFloat16:
        gpu_kernel_nocast(iter, CastScalarFunc<BFloat16, Float8_e5m2>());
        break;
      default:
        gpu_kernel(iter, CopyScalarFunc<Float8_e5m2>());
        break;
    }
  } else if (dtype == kFloat8_e4m3fnuz) {
    switch (other_dtype) {
      case kFloat:
        gpu_kernel_nocast(iter, CastScalarFunc<float, Float8_e4m3fnuz>());
        break;
      case kHalf:
        gpu_kernel_nocast(iter, CastScalarFunc<Half, Float8_e4m3fnuz>());
        break;
      case kBFloat16:
        gpu_kernel_nocast(iter, CastScalarFunc<BFloat16, Float8_e4m3fnuz>());
        break;
      default:
        gpu_kernel(iter, CopyScalarFunc<Float8_e4m3fnuz>());
        break;
    }
  } else if (dtype == kFloat8_e5m2fnuz) {
    switch (other_dtype) {
      case kFloat:
        gpu_kernel_nocast(iter, CastScalarFunc<float, Float8_e5m2fnuz>());
        break;
      case kHalf:
        gpu_kernel_nocast(iter, CastScalarFunc<Half, Float8_e5m2fnuz>());
        break;
      case kBFloat16:
        gpu_kernel_nocast(iter, CastScalarFunc<BFloat16, Float8_e5m2fnuz>());
        break;
      default:
        gpu_kernel(iter, CopyScalarFunc<Float8_e5m2fnuz>());
        break;
    }
  } else if (dtype == kFloat8_e8m0fnu) {
    switch (other_dtype) {
      case kFloat:
        gpu_kernel_nocast(iter, CastScalarFunc<float, Float8_e8m0fnu>());
        break;
      case kHalf:
        gpu_kernel_nocast(iter, CastScalarFunc<Half, Float8_e8m0fnu>());
        break;
      case kBFloat16:
        gpu_kernel_nocast(iter, CastScalarFunc<BFloat16, Float8_e8m0fnu>());
        break;
      default:
        gpu_kernel(iter, CopyScalarFunc<Float8_e8m0fnu>());
        break;
    }
  } else {
    TORCH_CHECK(
        false,
        "This input type is not Float8 type or has not been supported by copy.",
        dtype);
  }
}

void float4_copy_kernel_xpu(TensorIteratorBase& iter) {
  ScalarType src_dtype = iter.dtype(1);

  if (src_dtype == kFloat4_e2m1fn_x2) {
    gpu_kernel_nocast(iter, CopyScalarFunc<Float4_e2m1fn_x2>());
  } else {
    TORCH_CHECK(
        false,
        "Copy from ",
        src_dtype,
        " to Float4_e2m1fn_x2 has not been supported.");
  }
}

// Deliberately NOT CastScalarFunc. That template has a partial specialization
//
//     template <typename Float8DataType>
//     struct CastScalarFunc<Half, Float8DataType>
//
// whose second parameter is unconstrained despite the name, so
// CastScalarFunc<Half, float> would select it and pick up the float8
// workaround that maps -0.0 to +0.0 (IEEE makes `src_val == Half(-0.0)` true
// for both zeros). A plain fp16->fp32 cast must preserve the sign of zero, so
// this path uses its own functor.
template <typename in_scalar_t, typename out_scalar_t>
struct FpCastFunc {
  out_scalar_t operator()(in_scalar_t src_val) const {
    return (out_scalar_t)src_val;
  }
};

// fp32 <-> fp16 is by far the most common cast in practice, so it is worth a
// special case here.
//
// The generic path below reaches copy_kernel through gpu_kernel(), and any
// cross-dtype copy makes needs_dynamic_casting<>::check(iter) true. In
// Loops.h that diverts to gpu_kernel_impl()'s casting branch, which only ever
// calls launch_unrolled_kernel() -- launch_vectorized_kernel() exists solely
// inside gpu_kernel_impl_nocast(). The result is a scalar kernel:
//
//   unrolled_elementwise_kernel<CopyScalarFunc<c10::Half>, ...,
//                               LoadWithCast<1>, StoreWithCast<1>, 4>
//
// (the trailing 4 is item_work_size -- four scalar iterations per work-item,
// not a vector width).
//
// Doing the conversion inside the functor instead makes the functor's arg and
// result types match the iterator's dtypes exactly, so needs_dynamic_casting
// is false and gpu_kernel_nocast() can take the vectorized path. Same idea as
// float8_copy_kernel_xpu above.
//
// Measured on CRI (32 MiB LLC, 30 MiB resident working set, 6 B/element so
// both directions move identical traffic):
//
//            torch today   bare SYCL scalar   bare SYCL vec4
//   f32->f16     983 GB/s          1592 GB/s        4364 GB/s
//   f16->f32     852 GB/s          1627 GB/s        3308 GB/s
//
// Cost: two extra functor instantiations, each expanded over the vec_size
// switch in launch_vectorized_kernel (vec_size 1/2/4 for 4-byte elements;
// 8 is excluded by max_scalar_bytes * vec_size <= 16). That binary growth is
// why this is a named special case rather than something applied to every
// dtype pair.
void fp16_fp32_copy_kernel_xpu(TensorIteratorBase& iter) {
  ScalarType dst_dtype = iter.dtype(0);
  ScalarType src_dtype = iter.dtype(1);

  if (dst_dtype == kHalf && src_dtype == kFloat) {
    gpu_kernel_nocast(iter, FpCastFunc<float, Half>());
  } else if (dst_dtype == kFloat && src_dtype == kHalf) {
    gpu_kernel_nocast(iter, FpCastFunc<Half, float>());
  } else {
    TORCH_CHECK(
        false,
        "fp16_fp32_copy_kernel_xpu called with unsupported dtypes: src ",
        src_dtype,
        ", dst ",
        dst_dtype);
  }
}

void copy_kernel(TensorIteratorBase& iter) {
  ScalarType dtype = iter.common_dtype();
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_xpu", [&] {
      gpu_kernel(iter, CopyScalarFunc<scalar_t>());
    });
  } else if (isBitsType(dtype)) {
    TORCH_CHECK(
        iter.dtype(0) == iter.dtype(1),
        "copy_() does not support casting bits types to different bits types. "
        "Source dtype is ",
        iter.dtype(1),
        ", target dtype is ",
        iter.dtype(0));
    AT_DISPATCH_BIT_TYPES(dtype, "copy_xpu", [&] {
      gpu_kernel_nocast(iter, CopyScalarFunc<scalar_t>());
    });
  } else if (isFloat8Type(iter.dtype(0))) {
    float8_copy_kernel_xpu(iter);
  } else if (iter.dtype(0) == kFloat4_e2m1fn_x2) {
    float4_copy_kernel_xpu(iter);
  } else if (
      (iter.dtype(0) == kHalf && iter.dtype(1) == kFloat) ||
      (iter.dtype(0) == kFloat && iter.dtype(1) == kHalf)) {
    // Keyed on the actual pair, not common_dtype: for fp16<->fp32 the common
    // dtype promotes to Float either way, so it cannot tell the two
    // directions apart and would also match a same-dtype fp32 copy.
    fp16_fp32_copy_kernel_xpu(iter);
  } else {
    AT_DISPATCH_V2(
        dtype,
        "copy_xpu",
        AT_WRAP([&] { gpu_kernel(iter, CopyScalarFunc<scalar_t>()); }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        kHalf,
        kBool,
        kBFloat16,
        kComplexHalf,
        kBComplex32,
        AT_EXPAND(AT_FLOAT8_TYPES),
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

} // namespace at::native::xpu
