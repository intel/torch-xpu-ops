#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Sorting.h>

namespace at {

::std::tuple<Tensor, Tensor> XPUNativeFunctions::sort(
    const Tensor& self,
    ::std::optional<bool> stable,
    int64_t dim,
    bool descending) {
  Tensor values, indices;
  return native::xpu::sort_stable_kernel(
      self, stable, values, indices, dim, descending);
}

::std::tuple<Tensor&, Tensor&> XPUNativeFunctions::sort_out(
    const Tensor& self,
    ::std::optional<bool> stable,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  return native::xpu::sort_stable_kernel(
      self, stable, values, indices, dim, descending);
}

Tensor XPUNativeFunctions::argsort(
    const Tensor& self,
    bool stable,
    int64_t dim,
    bool descending) {
  Tensor values, indices;
  return std::get<1>(native::xpu::sort_stable_kernel(
      self, stable, values, indices, dim, descending));
}

} // namespace at
