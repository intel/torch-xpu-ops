#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/BucketizationKernels.h>

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

  // we have two inputs to set right, pre_check checks that they aren't set to
  // opposites
  bool is_right = (side_opt && *side_opt == "right") || right;
  if (self.numel() == 0) {
    return result;
  }

  // for non-contiguous result tensors, we write the output to a contiguous copy
  // so we can later copy back, maintaining the original result tensor
  Tensor out = result;
  if (!result.is_contiguous()) {
    out = result.contiguous();
  }
  if (sorted_sequence.is_contiguous() && self.is_contiguous() &&
      sorted_sequence.dtype() == self.dtype() && sorter.is_contiguous()) {
    at::native::xpu::searchsorted_kernel(
        out, self, sorted_sequence, out_int32, is_right, sorter);
  } else {
    Tensor trimmed_input;
    Tensor trimmed_boundaries;
    Tensor trimmed_sorter;
    at::native::searchsorted_maybe_trim_input_tensors(
        trimmed_input,
        trimmed_boundaries,
        trimmed_sorter,
        self,
        sorted_sequence,
        sorter);
    const Tensor& final_input = trimmed_input.defined() ? trimmed_input : self;
    const Tensor& final_boundaries =
        trimmed_boundaries.defined() ? trimmed_boundaries : sorted_sequence;
    const Tensor& final_sorter =
        trimmed_sorter.defined() ? trimmed_sorter : sorter;
    at::native::xpu::searchsorted_kernel(
        out, final_input, final_boundaries, out_int32, is_right, final_sorter);
  }

  // if result is non-contiguous, we wrote the answer to a copied version, so we
  // copy back to the original result tensor
  if (!result.is_contiguous()) {
    result.copy_(out);
  }
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
