#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include <aten/sycl/Resize.h>

namespace at {
namespace native {
namespace xpu {

const at::Tensor & resize_(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
    return native::xpu::resize_sycl_(self, size, memory_format); 
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::resize_"), TORCH_FN(resize_));
}

}}} // at::native::xpu
