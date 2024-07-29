#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
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

static Tensor& masked_select_out_impl(
    Tensor& result,
    const Tensor& self,
    const Tensor& mask) {
  NoNamesGuard guard;

  TORCH_CHECK(
      mask.scalar_type() == ScalarType::Bool,
      "masked_select: expected BoolTensor for mask");
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "masked_select(): self and result must have the same scalar type");

  auto mask_temp = (mask.dim() == 0)
      ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0))
      : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp = (self.dim() == 0)
      ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0))
      : c10::MaybeOwned<Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto mask_self_expanded = expand_outplace(*mask_temp, *self_temp);
  XPUNativeFunctions::index_out(
      *std::get<1>(mask_self_expanded),
      c10::List<std::optional<at::Tensor>>(
          {*std::move(std::get<0>(mask_self_expanded))}),
      result);

  return result;
}

Tensor XPUNativeFunctions::masked_select(
    const Tensor& self,
    const Tensor& mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  Tensor result = at::empty({0}, self.options());
  return masked_select_out_impl(result, self, mask);
}

Tensor& XPUNativeFunctions::masked_select_out(
    const Tensor& self,
    const Tensor& mask,
    Tensor& result) {
  namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_impl(result, self, mask);
}

} // namespace at
