#pragma once

#include <ATen/core/Tensor.h>
#include <aten/EmptyTensor.h>

namespace at{
namespace xpu {

Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  if (strides.empty()) {
      return at::detail::empty_xpu(sizes, options);
  } else {
      return at::detail::empty_strided_xpu(sizes, strides, options);
  }
}

void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  TORCH_CHECK(options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(),
      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
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
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}

}
}
