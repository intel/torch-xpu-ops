#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>

namespace at::native::xpu {

template <typename scalar_t>
void aminmax_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      MinMaxOps<scalar_t, scalar_t, int32_t>{},
      std::pair<scalar_t, scalar_t>(
          at::numeric_limits<scalar_t>::upper_bound(),
          at::numeric_limits<scalar_t>::lower_bound()));
}

void aminmax_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      at::ScalarType::Bool,
      iter.input_dtype(),
      "aminmax_xpu",
      [&]() { aminmax_kernel_impl<scalar_t>(iter); });
}

} // namespace at::native::xpu
