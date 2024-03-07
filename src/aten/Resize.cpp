#include <ATen/core/Tensor.h>
#include <ATen/native/mps/Copy.h>
#include <torch/library.h>

#include <aten/sycl/Resize.h>

namespace at {
namespace native {
namespace xpu {

const at::Tensor & resize_(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
    return native::xpu::resize_sycl_(self, size, memory_format); 
}

Tensor _copy_from_and_resize_mps(const at::Tensor& self, const at::Tensor& dst) {
  return at::native::mps::mps_copy_(const_cast<Tensor&>(dst), self, false);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::resize_"), TORCH_FN(resize_));
  m.impl(TORCH_SELECTIVE_NAME("aten::_copy_from_and_resize"), TORCH_FN(_copy_from_and_resize_mps));
}

}}} // at::native::xpu
