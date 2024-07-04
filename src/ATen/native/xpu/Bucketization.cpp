#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/BucketizationKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

Tensor& XPUNativeFunctions::searchsorted_out(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const std::optional<c10::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt,
    Tensor& result) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> sorter_maybe_owned =
      at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  at::native::searchsorted_pre_check(
      sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  at::native::resize_output(result, self.sizes());

  if (self.numel() == 0) {
    return result;
  }

  // we have two inputs to set right, pre_check checks that they aren't set to
  // opposites
  bool is_right = (side_opt && *side_opt == "right") || right;
  at::native::xpu::searchsorted_kernel(
      result, self, sorted_sequence, out_int32, is_right, sorter);
  return result;
}

Tensor& XPUNativeFunctions::searchsorted_out(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const std::optional<c10::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt,
    Tensor& result) {
  const Tensor& scalar_tensor =
      at::native::searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted_out(
      sorted_sequence,
      scalar_tensor,
      out_int32,
      right,
      side_opt,
      sorter_opt,
      result);
}

Tensor XPUNativeFunctions::searchsorted(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const std::optional<c10::string_view> side_opt,
    const std::optional<Tensor>& sorter) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  searchsorted_out(
      sorted_sequence, self, out_int32, right, side_opt, sorter, result);
  return result;
}

Tensor XPUNativeFunctions::searchsorted(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const std::optional<c10::string_view> side_opt,
    const std::optional<Tensor>& sorter) {
  const Tensor& scalar_tensor =
      at::native::searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted(
      sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter);
}

Tensor& XPUNativeFunctions::bucketize_out(
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    Tensor& result) {
  TORCH_CHECK(
      boundaries.dim() == 1,
      "boundaries tensor must be 1 dimension, but got dim(",
      boundaries.dim(),
      ")");
  searchsorted_out(
      boundaries, self, out_int32, right, nullopt, nullopt, result);
  return result;
}

Tensor XPUNativeFunctions::bucketize(
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  bucketize_out(self, boundaries, out_int32, right, result);
  return result;
}

Tensor XPUNativeFunctions::bucketize(
    const Scalar& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  return bucketize(
      at::native::searchsorted_scalar_tensor(self, boundaries.device()),
      boundaries,
      out_int32,
      right);
}
} // namespace at
