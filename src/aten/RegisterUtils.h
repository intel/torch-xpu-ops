#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <aten/EmptyTensor.h>

namespace at {
namespace xpu {

inline Tensor create_out(
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  if (strides.empty()) {
    return at::detail::empty_xpu(sizes, options);
  } else {
    return at::detail::empty_strided_xpu(sizes, strides, options);
  }
}

inline void resize_out(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  TORCH_CHECK(
      options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ",
      options.dtype(),
      ", but got ",
      out.dtype(),
      " instead");
  TORCH_CHECK(
      options.device() == out.device(),
      "Expected out tensor to have device ",
      options.device(),
      ", but got ",
      out.device(),
      " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(
          *options.memory_format_opt());
    }
  }
}

inline Tensor& resize_reduction(
    Tensor& out,
    const Tensor& self,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType out_dtype,
    bool allow_empty_dims = false) {
  DimVector dims_ = at::native::make_dim_vector(opt_dims, self.dim());
  maybe_wrap_dims(dims_, self.dim());
  auto shape =
      at::meta::get_reduction_shape(self, dims_, keepdim, allow_empty_dims);
  if (self.layout() == kStrided) {
    if (out.defined()) {
      resize_out(out, shape, {}, self.options().dtype(out_dtype));
    } else {
      out = create_out(shape, {}, self.options().dtype(out_dtype));
    }
  } else if (shape.size() == 0) {
    if (out.defined()) {
      resize_out(
          out, shape, {}, self.options().dtype(out_dtype).layout(kStrided));
    } else {
      out = create_out(
          shape, {}, self.options().dtype(out_dtype).layout(kStrided));
    }
  } else {
    TORCH_CHECK(
        false,
        "resize_reduction: support for output with ",
        self.layout(),
        " layout is not implemented yet");
  }
  namedinference::propagate_names_for_reduction(out, self, dims_, keepdim);
  return out;
}

inline std::tuple<Tensor&, Tensor&> resize_reduction_with_indices(
    Tensor& out,
    Tensor& out_indice,
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    ScalarType out_dtype) {
  DimVector dims_(dims);
  maybe_wrap_dims(dims_, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_, keepdim);
  if (out.defined()) {
    resize_out(out, shape, {}, self.options().dtype(out_dtype));
  } else {
    out = create_out(shape, {}, self.options().dtype(out_dtype));
  }
  if (out_indice.defined()) {
    resize_out(out_indice, shape, {}, self.options().dtype(kLong));
  } else {
    out_indice = create_out(shape, {}, self.options().dtype(kLong));
  }
  namedinference::propagate_names_for_reduction(out, self, dims_, keepdim);
  namedinference::propagate_names_for_reduction(
      out_indice, self, dims_, keepdim);
  return std::forward_as_tuple(out, out_indice);
}

} // namespace xpu
} // namespace at
