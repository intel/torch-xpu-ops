#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <aten/sycl/FillKernel.h>

namespace at {
namespace native {
namespace xpu {

Tensor& fill_out(Tensor& self, const Scalar& value) {
  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(
                      false) // Fill is idempotent, so overlap is okay
                  .check_all_same_dtype(false)
                  .add_output(self)
                  .resize_outputs(false)
                  .build();
  native::xpu::fill_kernel(iter, value);
  return self;
}

Tensor& fill_scalar_(Tensor& self, const Scalar& value) {
  return fill_out(self, value);
}

Tensor& zero_(Tensor& self) {
  return self.fill_(0);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::fill_.Scalar"), TORCH_FN(fill_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::zero_"), TORCH_FN(zero_));
}

} // namespace xpu
} // namespace native
} // namespace at
