#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>

namespace at {

TensorIterator addcmul_meta(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value,
    Tensor& out) {
  auto iter = at::TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(out)
                  .add_input(self)
                  .add_input(tensor1)
                  .add_input(tensor2)
                  .build();
  return iter;
}

Tensor& XPUNativeFunctions::addcmul_out(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value,
    Tensor& out) {
  auto iter = addcmul_meta(self, tensor1, tensor2, value, out);
  native::xpu::addcmul_kernel(iter, value);
  return out;
}

Tensor XPUNativeFunctions::addcmul(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  Tensor out;
  auto iter = addcmul_meta(self, tensor1, tensor2, value, out);
  native::xpu::addcmul_kernel(iter, value);
  return iter.output();
}

Tensor& XPUNativeFunctions::addcmul_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  auto iter = addcmul_meta(self, tensor1, tensor2, value, self);
  native::xpu::addcmul_kernel(iter, value);
  return self;
}

} // namespace at
