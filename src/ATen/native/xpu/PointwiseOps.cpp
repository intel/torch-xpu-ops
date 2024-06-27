#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>

namespace at {

TensorIterator addcdiv_meta(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value,
    Tensor& out) {
  if (isIntegralType(tensor1.scalar_type(), /*includeBool=*/true) &&
      isIntegralType(tensor2.scalar_type(), /*includeBool=*/true)) {
    TORCH_CHECK(
        false,
        "Integer division with addcdiv is no longer supported, and in a future  ",
        "release addcdiv will perform a true division of tensor1 and tensor2. ",
        "The historic addcdiv behavior can be implemented as ",
        "(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) ",
        "for integer inputs and as ",
        "(input + value * tensor1 / tensor2) for float inputs. ",
        "The future addcdiv behavior is just the latter implementation: ",
        "(input + value * tensor1 / tensor2), for all dtypes.");
  }

  TensorIterator iter;
  iter.build_ternary_op(out, self, tensor1, tensor2);
  return iter;
}

Tensor& XPUNativeFunctions::addcdiv_out(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value,
    Tensor& out) {
  auto iter = addcdiv_meta(self, tensor1, tensor2, value, out);
  native::xpu::addcdiv_kernel(iter, value);
  return out;
}

Tensor XPUNativeFunctions::addcdiv(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  Tensor out;
  auto iter = addcdiv_meta(self, tensor1, tensor2, value, out);
  native::xpu::addcdiv_kernel(iter, value);
  return iter.output();
}

Tensor& XPUNativeFunctions::addcdiv_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value) {
  auto iter = addcdiv_meta(self, tensor1, tensor2, value, self);
  native::xpu::addcdiv_kernel(iter, value);
  return self;
}

TensorIterator addcmul_meta(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value,
    Tensor& out) {
  TensorIterator iter;
  iter.build_ternary_op(out, self, tensor1, tensor2);
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
