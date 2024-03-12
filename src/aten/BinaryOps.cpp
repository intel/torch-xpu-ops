#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>

#include <aten/sycl/BinaryKernels.h>

namespace at {

at::Tensor XPUNativeFunctions::add(
    const at::Tensor & self,
    const at::Tensor & other,
    const at::Scalar & alpha) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return iter.output();
}

at::Tensor & XPUNativeFunctions::add_(
    at::Tensor & self,
    const at::Tensor & other,
    const at::Scalar & alpha) {
  auto iter = TensorIterator::binary_op(self, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return self;
}

at::Tensor XPUNativeFunctions::add(
    const at::Tensor & self,
    const at::Scalar & other,
    const at::Scalar & alpha) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::add(self, wrapper, alpha);
}

at::Tensor & XPUNativeFunctions::add_(
    at::Tensor & self,
    const at::Scalar & other,
    const at::Scalar & alpha) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::add_(self, wrapper, alpha);
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

Tensor XPUNativeFunctions::mul(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::mul(self, wrapper);
}

Tensor& XPUNativeFunctions::mul_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::mul_(self, wrapper);
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

Tensor XPUNativeFunctions::div(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::div(self, wrapper);
}

Tensor& XPUNativeFunctions::div_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::div_(self, wrapper);
}

} // at
