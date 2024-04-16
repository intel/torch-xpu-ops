#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>

#include <aten/sycl/IndexingKernel.h>
#include <comm/TensorInfo.h>

namespace at {

Tensor XPUNativeFunctions::index(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices) {
  TORCH_CHECK(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");

  check_indices_on_cpu_or_selfdevice(self, indices);
  auto info = make_info(self, indices);
  auto iter = make_index_iterator(info);
  native::xpu::index_kernel(
      iter,
      info.indexed_sizes,
      info.indexed_strides,
      info.non_indexed_sizes,
      info.non_indexed_strides);
  return iter.output();
}

Tensor& XPUNativeFunctions::index_select_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::index_select_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::index_select_out", "index");
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::index_select_out", "out");

  static constexpr string_view DIM_WARNING =
      "Tensor too large or too many (> 12) dimensions";
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

  dim = at::maybe_wrap_dim(dim, self);
  TORCH_CHECK(self.dim() <= XPU_MAX_TENSORINFO_DIMS, DIM_WARNING);
  TORCH_CHECK(index.dim() <= XPU_MAX_TENSORINFO_DIMS, DIM_WARNING);
  native::xpu::index_select_kernel(self, dim, index, out);

  return out;
}

Tensor XPUNativeFunctions::index_select(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  auto out = at::empty({0}, self.options());
  return index_select_out(self, dim, index, out);
}
} // namespace at
