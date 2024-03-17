#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/ActivationOpsKernels.h>

namespace at {

Tensor& XPUNativeFunctions::relu_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::relu_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::threshold_out(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, self);
  native::xpu::threshold_kernel(iter, threshold, value);
  return out;
}

Tensor& XPUNativeFunctions::threshold_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
  auto iter = TensorIterator::binary_op(grad_input, self, grad_output);
  native::xpu::threshold_kernel(iter, threshold, 0);
  return grad_input;
}

Tensor& XPUNativeFunctions::gelu_out(
    const Tensor& self,
    c10::string_view approximate,
    Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::gelu_kernel(iter, approximate);
  return out;
}

Tensor& XPUNativeFunctions::gelu_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    c10::string_view approximate,
    Tensor& grad_input) {
  auto iter = TensorIterator::binary_op(grad_input, grad_output, self);
  native::xpu::gelu_backward_kernel(iter, approximate);
  return grad_input;
}

Tensor& XPUNativeFunctions::tanh_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  auto iter = TensorIterator::binary_op(grad_input, grad_output, output);
  native::xpu::tanh_backward_kernel(iter);
  return grad_input;
}

} // namespace at
