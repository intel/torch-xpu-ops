#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/BinaryKernels.h>

namespace at {

Tensor XPUNativeFunctions::add(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return iter.output();
}

Tensor& XPUNativeFunctions::add_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  auto iter = TensorIterator::binary_op(self, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return self;
}

Tensor& XPUNativeFunctions::add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return out;
}

Tensor XPUNativeFunctions::add(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::add(self, wrapper, alpha);
}

Tensor& XPUNativeFunctions::add_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::add_(self, wrapper, alpha);
}

Tensor& XPUNativeFunctions::add_out(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::add_out(self, wrapper, alpha, out);
}

Tensor XPUNativeFunctions::sub(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  Tensor out;
  native::sub_check(self, other);
  auto iter = TensorIterator::binary_op(out, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::sub_kernel(iter, alpha);
  return iter.output();
}

Tensor& XPUNativeFunctions::sub_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  native::sub_check(self, other);
  auto iter = TensorIterator::binary_op(self, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::sub_kernel(iter, alpha);
  return self;
}

Tensor& XPUNativeFunctions::sub_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  native::sub_check(self, other);
  auto iter = TensorIterator::binary_op(out, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::sub_kernel(iter, alpha);
  return out;
}

Tensor XPUNativeFunctions::sub(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::sub(self, wrapper, alpha);
}

Tensor& XPUNativeFunctions::sub_(
    Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::sub_(self, wrapper, alpha);
}

Tensor& XPUNativeFunctions::sub_out(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::sub_out(self, wrapper, alpha, out);
}

Tensor XPUNativeFunctions::mul(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, self, other);
  native::xpu::mul_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::mul_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  native::xpu::mul_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::mul_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  native::xpu::mul_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::mul(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::mul(self, wrapper);
}

Tensor& XPUNativeFunctions::mul_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::mul_(self, wrapper);
}

Tensor& XPUNativeFunctions::mul_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::mul_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::div(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::binary_float_op(out, self, other);
  native::xpu::div_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::div_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_float_op(self, self, other);
  native::xpu::div_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::div_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_float_op(out, self, other);
  native::xpu::div_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::div(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::div(self, wrapper);
}

Tensor& XPUNativeFunctions::div_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::div_(self, wrapper);
}

Tensor& XPUNativeFunctions::div_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::div_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::rsub(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  return XPUNativeFunctions::sub(self, other, alpha);
}

Tensor& XPUNativeFunctions::rsub_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  return XPUNativeFunctions::sub_out(self, other, alpha, out);
}

Tensor XPUNativeFunctions::rsub(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha) {
  return XPUNativeFunctions::sub(
      self, native::wrapped_scalar_tensor(other), alpha);
}

Tensor& XPUNativeFunctions::rsub_out(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha,
    Tensor& out) {
  return XPUNativeFunctions::sub_out(
      self, native::wrapped_scalar_tensor(other), alpha, out);
}

Tensor& XPUNativeFunctions::remainder_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  native::xpu::remainder_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::fmod_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  native::xpu::fmod_kernel(iter);
  return out;
}

} // namespace at
