#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <aten/sycl/UnaryKernels.h>

namespace at {
namespace native {
namespace xpu {

Tensor& abs_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::abs_kernel(iter);
  return out;
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::abs.out"), TORCH_FN(abs_out));
}

} // namespace xpu
} // namespace native
} // namespace at
