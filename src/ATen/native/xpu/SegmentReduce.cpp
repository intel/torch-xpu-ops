#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReductionType.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/SegmentReduceKernels.h>

namespace at {

Tensor XPUNativeFunctions::segment_reduce(
    const Tensor& data,
    c10::string_view reduce,
    const std::optional<Tensor>& lengths,
    const std::optional<Tensor>& indices,
    const std::optional<Tensor>& offsets,
    int64_t axis,
    bool unsafe,
    const std::optional<Scalar>& initial) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  TORCH_CHECK(data.numel() >= 0);

  // check that one of lengths or offsets is defined
  auto lengths_has_value = lengths.has_value();
  auto offsets_has_value = offsets.has_value();
  TORCH_CHECK(
      !indices.has_value(),
      "segment_reduce(): indices based reduction is not supported yet.");
  TORCH_CHECK(
      lengths_has_value || offsets_has_value,
      "segment_reduce(): Either lengths or offsets must be defined.")

  auto reduction = native::get_reduction_enum(reduce);
  const auto data_contig = data.contiguous();

  if (offsets_has_value) {
    const auto& offsets_value = offsets.value();

    // offsets related checks
    TORCH_CHECK(data.get_device() == offsets_value.get_device());
    TORCH_CHECK(data.dim() >= offsets_value.dim());
    TORCH_CHECK(
        axis == offsets_value.dim() - 1,
        "segment_reduce(): Expected axis to be the last dimension of offsets but got ",
        axis,
        ".");

    // TODO: add checks when !unsafe

    const auto offsets_contig = offsets_value.contiguous();

    return native::xpu::_segment_reduce_offsets_kernel(
        reduction, data_contig, offsets_contig, axis, initial);

  } else {
    const auto& lengths_value = lengths.value();

    // length related checks
    TORCH_CHECK(data.get_device() == lengths_value.get_device());
    TORCH_CHECK(data.dim() >= lengths_value.dim());
    TORCH_CHECK(
        axis == lengths_value.dim() - 1,
        "segment_reduce(): Expected axis to be the last dimension of lengths but got ",
        axis,
        ".");

    if (!unsafe) {
      auto min_length = lengths_value.min().item<int64_t>();
      TORCH_CHECK((min_length >= 0), "lengths contains negative value!");
      TORCH_CHECK(
          all(lengths_value.sum({-1}) == data.size(axis)).item<bool>(),
          "segment_reduce(): Expected all rows of lengths along axis ",
          "to sum to data.size(lengths.dim()-1) when !unsafe.");
    }

    const auto lengths_contig = lengths_value.contiguous();

    return native::xpu::_segment_reduce_lengths_kernel(
        reduction, data_contig, lengths_contig, axis, initial);
  }
}

Tensor XPUNativeFunctions::_segment_reduce_backward(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    c10::string_view reduce,
    const std::optional<Tensor>& lengths,
    const std::optional<Tensor>& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  // check that one of lengths or offsets is defined
  // codegen for derivatives.yaml passes an undefined Tensor for None rather
  // than a std::optional so checking .has_value() doesn't work unlike in the
  // forward pass
  auto lengths_has_value = lengths.has_value() && lengths.value().defined();
  auto offsets_has_value = offsets.has_value() && offsets.value().defined();
  TORCH_CHECK(
      lengths_has_value || offsets_has_value,
      "segment_reduce(): Either lengths or offsets must be defined.");

  const auto grad_contig = grad.contiguous();
  const auto output_contig = output.contiguous();
  const auto data_contig = data.contiguous();
  auto reduction = native::get_reduction_enum(reduce);

  if (offsets_has_value) {
    const auto& offsets_value = offsets.value();
    const auto offsets_contig = offsets_value.contiguous();
    return native::xpu::_segment_reduce_offsets_backward_kernel(
        grad_contig,
        output_contig,
        data_contig,
        reduction,
        offsets_contig,
        axis,
        initial);
  } else {
    const auto& lengths_value = lengths.value();
    const auto lengths_contig = lengths_value.contiguous();
    return native::xpu::_segment_reduce_lengths_backward_kernel(
        grad_contig,
        output_contig,
        data_contig,
        reduction,
        lengths_contig,
        axis,
        initial);
  }
}

} // namespace at
