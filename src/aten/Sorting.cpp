#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
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
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, values, "xpu::sort_out_values_stable", "values");
  c10::impl::check_and_update_common_device(
      common_device, indices, "xpu::sort_out_values_stable", "indices");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::sort_out_values_stable", "self");
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
