#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Sorting.h>
#include <comm/RegisterUtils.h>

namespace at {

void sort_stable_meta(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  maybe_wrap_dim(dim, self.dim());

  // See issue: https://github.com/pytorch/pytorch/issues/65863
  // Strides should be dense, so as not to allocate too much memory.
  // We either use 'self' strides, or infer dense strides from them.
  std::vector<int64_t> strides = (self.is_non_overlapping_and_dense())
      ? self.strides().vec()
      : at::infer_dense_strides(self.sizes(), self.strides());
  auto sizes = self.sizes();
  if (values.defined()) {
    at::xpu::resize_out(values, sizes, strides, self.options());
  } else {
    values = at::xpu::create_out(sizes, strides, self.options());
  }
  if (indices.defined()) {
    at::xpu::resize_out(indices, sizes, strides, self.options().dtype(kLong));
  } else {
    indices = at::xpu::create_out(sizes, strides, self.options().dtype(kLong));
  }
}

::std::tuple<Tensor, Tensor> XPUNativeFunctions::sort(
    const Tensor& self,
    ::std::optional<bool> stable,
    int64_t dim,
    bool descending) {
  Tensor values, indices;
  sort_stable_meta(self, values, indices, dim);
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
  sort_stable_meta(self, values, indices, dim);
  return native::xpu::sort_stable_kernel(
      self, stable, values, indices, dim, descending);
}

Tensor XPUNativeFunctions::argsort(
    const Tensor& self,
    bool stable,
    int64_t dim,
    bool descending) {
  Tensor values, indices;
  sort_stable_meta(self, values, indices, dim);
  return std::get<1>(native::xpu::sort_stable_kernel(
      self, stable, values, indices, dim, descending));
}

} // namespace at
