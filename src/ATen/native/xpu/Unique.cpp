#include <ATen/native/xpu/sycl/UniqueKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <iostream>

namespace at {

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::unique_consecutive(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    std::optional<int64_t> dim) {
  std::cout << "zhy--- call unique_consecutive. line" << __LINE__ << std::endl;
  if (!dim.has_value()) {
    std::cout << "zhy--- call unique_consecutive. line" << __LINE__
              << std::endl;
    return native::xpu::unique_consecutive_kernel(
        self, return_inverse, return_counts, dim);
  }
  return native::xpu::unique_dim_consecutive_kernel(
      self, dim.value(), return_inverse, return_counts);
}

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::unique_dim_consecutive(
    const at::Tensor& self,
    int64_t dim,
    bool return_inverse,
    bool return_counts) {
  std::cout << "zhy--- call unique_dim_consecutive. line" << __LINE__
            << std::endl;
  return native::xpu::unique_dim_consecutive_kernel(
      self, dim, return_inverse, return_counts);
}

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::unique_dim(
    const Tensor& self,
    const int64_t dim,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  return native::xpu::unique_dim_kernel(
      self, dim, return_inverse, return_counts);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::_unique(
    const Tensor& self,
    const bool sorted,
    const bool return_inverse) {
  std::cout << "zhy--- call unique. line" << __LINE__ << std::endl;
  return native::xpu::_unique_kernel(self, return_inverse);
}

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::_unique2(
    const Tensor& self,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  std::cout << "zhy--- call unique2. line" << __LINE__ << std::endl;
  return native::xpu::_unique2_kernel(self, return_inverse, return_counts);
}

} // namespace at