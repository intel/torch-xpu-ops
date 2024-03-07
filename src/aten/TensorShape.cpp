#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/as_strided_ops.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace xpu {

Tensor view(const Tensor& self, IntArrayRef size) {
  return at::native::view(self, size);
}

Tensor as_strided(
    const Tensor & self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset=c10::nullopt) {
  if (self.is_quantized()) {
    return at::native::as_strided_qtensorimpl(self, size, stride, storage_offset);
  }
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
  m.impl(TORCH_SELECTIVE_NAME("aten::as_strided"), TORCH_FN(as_strided));
}

}}} // namespace at::native_xpu
