#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/FillKernel.h>

namespace at {

Tensor& XPUNativeFunctions::fill_(Tensor& self, const Scalar& value) {
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

Tensor& XPUNativeFunctions::zero_(Tensor& self) {
  return self.fill_(0);
}

} // namespace at
