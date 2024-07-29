#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/IndexingKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/TensorInfo.h>

namespace at {

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

Tensor& XPUNativeFunctions::masked_scatter_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter_: expected self and source to have same dtypes but got ",
      self.scalar_type(),
      " and ",
      source.scalar_type());
  TORCH_CHECK(
      mask.dtype() == ScalarType::Bool,
      "masked_scatter_ only supports boolean masks, "
      "but got mask with dtype ",
      mask.dtype());

  c10::MaybeOwned<Tensor> b_mask =
      expand_inplace(self, mask, "masked_scatter_");

  if (self.numel() == 0) {
    return self;
  }

  auto maskPrefixSum = at::empty(self.sizes(), mask.options().dtype(kLong));
  native::xpu::masked_scatter_kernel(self, *b_mask, maskPrefixSum, source);

  return self;
}

} // namespace at
