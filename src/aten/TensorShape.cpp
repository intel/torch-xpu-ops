#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/as_strided_ops.h>
#include <ATen/XPUNativeFunctions.h>

namespace at {

Tensor XPUNativeFunctions::view(const Tensor& self, IntArrayRef size) {
  return at::native::view(self, size);
}

Tensor XPUNativeFunctions::view_as_real(const at::Tensor& self) {
  return at::native::view_as_real(self);
}

Tensor XPUNativeFunctions::view_as_complex(const Tensor& self) {
  return at::native::view_as_complex(self);
}

Tensor XPUNativeFunctions::as_strided(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset = c10::nullopt) {
  if (self.is_quantized()) {
    return at::native::as_strided_qtensorimpl(
        self, size, stride, storage_offset);
  }
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

Tensor XPUNativeFunctions::_reshape_alias(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride) {
  return at::native::_reshape_alias(self, size, stride);
}

Tensor XPUNativeFunctions::unfold(
    const Tensor& self,
    int64_t dimension,
    int64_t size,
    int64_t step) {
  return at::native::unfold(self, dimension, size, step);
}

} // namespace at
