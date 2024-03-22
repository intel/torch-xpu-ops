#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/ReduceMaxValuesKernel.h>
#include <aten/sycl/ReduceMinValuesKernel.h>
#include <torch/library.h>

#include <iostream>

namespace at {

Tensor& XPUNativeFunctions::max_out(const Tensor& self, Tensor& out) {
  auto iter = at::native::make_reduction(
      "max_all", out, self, IntArrayRef{}, false, self.scalar_type());
  at::native::xpu::max_all_launch_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::max(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  XPUNativeFunctions::max_out(self, result);
  return result;
}

::std::tuple<Tensor&, Tensor&> XPUNativeFunctions::max_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& max,
    Tensor& max_values) {
  auto iter = meta::make_reduction(
      self, max, max_values, dim, keepdim, self.scalar_type(), kLong);
  at::native::xpu::max_launch_kernel(iter);
  return {max, max_values};
}

Tensor& XPUNativeFunctions::min_out(const Tensor& self, Tensor& out) {
  auto iter = at::native::make_reduction(
      "min_all", out, self, IntArrayRef{}, false, self.scalar_type());
  at::native::xpu::min_all_launch_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::min(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  XPUNativeFunctions::min_out(self, result);
  return result;
}

::std::tuple<Tensor&, Tensor&> XPUNativeFunctions::min_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& min,
    Tensor& min_indices) {
  auto iter = meta::make_reduction(
      self, min, min_indices, dim, keepdim, self.scalar_type(), kLong);
  at::native::xpu::min_launch_kernel(iter);
  return {min, min_indices};
}

} // namespace at
