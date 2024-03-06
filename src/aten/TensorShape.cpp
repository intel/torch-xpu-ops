#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace xpu {

Tensor view(const Tensor& self, IntArrayRef size) {
  return at::native::view(self, size);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::view"), TORCH_FN(view));
}

}}} // namespace at::native_xpu
