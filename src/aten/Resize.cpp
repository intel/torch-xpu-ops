#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include <aten/sycl/Resize.h>
#include <aten/Copy.h>

namespace at {
namespace native {
namespace xpu {

const at::Tensor & resize_(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
    return resize_sycl_(self, size, memory_format); 
}

Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  return copy_xpu(const_cast<Tensor&>(dst), self, false);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::resize_"), TORCH_FN(resize_));
  m.impl(TORCH_SELECTIVE_NAME("aten::_copy_from_and_resize"), TORCH_FN(_copy_from_and_resize));
}

}}} // at::native::xpu
