#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/BinaryBitwiseOpsKernels.h>

namespace at {

Tensor& XPUNativeFunctions::bitwise_and_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  native::xpu::bitwise_and_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::bitwise_or_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  native::xpu::bitwise_or_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::bitwise_xor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  native::xpu::bitwise_xor_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::bitwise_not_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::bitwise_not_kernel(iter);
  return out;
}

} // namespace at
