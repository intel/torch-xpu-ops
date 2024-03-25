#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct CopyScalarFunc {
  scalar_t operator()(scalar_t src_val) const {
    return src_val;
  }
};

void copy_kernel(TensorIterator& iter) {
  ScalarType dtype = iter.common_dtype();
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_xpu", [&] {
      gpu_kernel(iter, CopyScalarFunc<scalar_t>());
    });
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
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES),
        kFloat8_e4m3fn,
        kFloat8_e5m2);
  }
}

} // namespace at::native::xpu
