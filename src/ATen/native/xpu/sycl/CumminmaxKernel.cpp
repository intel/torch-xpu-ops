#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/ScanKernels.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>

namespace at::native::xpu {

void launch_cummax_kernel(
    const Tensor& self,
    const Tensor& values,
    const Tensor& indices,
    int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool,
      ScalarType::Half,
      ScalarType::BFloat16,
      self.scalar_type(),
      "cummax_xpu",
      [&]() {
        scalar_t init = self.is_floating_point()
            ? -std::numeric_limits<scalar_t>::infinity()
            : std::numeric_limits<scalar_t>::lowest();
        scan_with_indices<INCLUSIVE_TYPE, scalar_t, scalar_t, int64_t>(
            self, values, indices, dim, init, std::greater_equal<scalar_t>());
      });
}

void launch_cummin_kernel(
    const Tensor& self,
    const Tensor& values,
    const Tensor& indices,
    int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool,
      ScalarType::Half,
      ScalarType::BFloat16,
      self.scalar_type(),
      "cummin_xpu",
      [&]() {
        scalar_t init = self.is_floating_point()
            ? std::numeric_limits<scalar_t>::infinity()
            : std::numeric_limits<scalar_t>::max();
        scan_with_indices<INCLUSIVE_TYPE, scalar_t, scalar_t, int64_t>(
            self, values, indices, dim, init, std::less_equal<scalar_t>());
      });
}

} // namespace at::native::xpu
