#include <comm/xpu_aten.h>

#include <ATen/Dispatch.h>
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
template <>
struct CastScalarFunc<Half, Float8_e4m3fn> {
  C10_HOST_DEVICE Float8_e4m3fn operator()(Half src_val) const {
    return Float8_e4m3fn(c10::detail::fp16_ieee_to_fp32_value(src_val.x));
  }
};

template <>
struct CastScalarFunc<Half, Float8_e4m3fnuz> {
  C10_HOST_DEVICE Float8_e4m3fnuz operator()(Half src_val) const {
    return Float8_e4m3fnuz(c10::detail::fp16_ieee_to_fp32_value(src_val.x));
  }
};

template <>
struct CastScalarFunc<Half, Float8_e5m2> {
  C10_HOST_DEVICE Float8_e5m2 operator()(Half src_val) const {
    return Float8_e5m2(c10::detail::fp16_ieee_to_fp32_value(src_val.x));
  }
};

template <>
struct CastScalarFunc<Half, Float8_e5m2fnuz> {
  C10_HOST_DEVICE Float8_e5m2fnuz operator()(Half src_val) const {
    return Float8_e5m2fnuz(c10::detail::fp16_ieee_to_fp32_value(src_val.x));
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
    TORCH_CHECK(false, "Copy from ", src_dtype, " to Float4_e2m1fn_x2 has not been supported.");
  }
}

void copy_kernel(TensorIteratorBase& iter) {
  ScalarType dtype = iter.common_dtype();
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_xpu", [&] {
      gpu_kernel(iter, CopyScalarFunc<scalar_t>());
    });
  } else if (isFloat8Type(iter.dtype(0))) {
    float8_copy_kernel_xpu(iter);
  } else if (iter.dtype(0) == kFloat4_e2m1fn_x2) {
    float4_copy_kernel_xpu(iter);
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
        AT_EXPAND(AT_FLOAT8_TYPES),
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

} // namespace at::native::xpu
