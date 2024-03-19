#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/PowKernels.h>

namespace at {

Tensor XPUNativeFunctions::pow(const Tensor& self, const Tensor& exponent) {
  Tensor out;
  auto iter = TensorIterator::binary_op(out, self, exponent);
  native::xpu::pow_tensor_tensor_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::pow_(Tensor& self, const Tensor& exponent) {
  auto iter = TensorIterator::binary_op(self, self, exponent);
  native::xpu::pow_tensor_tensor_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::pow_out(
    const Tensor& base,
    const Tensor& exp,
    Tensor& out) {
  auto iter = TensorIterator::binary_op(out, base, exp);
  native::xpu::pow_tensor_tensor_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::pow(const Tensor& self, const Scalar& exponent) {
  Tensor out;
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::pow_tensor_scalar_kernel(iter, exponent);
  return iter.output();
}

Tensor& XPUNativeFunctions::pow_(Tensor& self, const Scalar& exponent) {
  if (exponent.equal(0.0) || exponent.equal(false)) {
    self.fill_(1);
  } else if (exponent.equal(1.0) || exponent.equal(true)) {
  } else {
    auto iter = TensorIterator::unary_op(self, self);
    native::xpu::pow_tensor_scalar_kernel(iter, exponent);
  }
  return self;
}

Tensor& XPUNativeFunctions::pow_out(
    const Tensor& self,
    const Scalar& exponent,
    Tensor& out) {
  if (exponent.equal(0.0) || exponent.equal(false)) {
    out.fill_(1);
  } else if (exponent.equal(1.0) || exponent.equal(true)) {
    out.copy_(self);
  } else {
    auto iter = TensorIterator::unary_op(out, self);
    native::xpu::pow_tensor_scalar_kernel(iter, exponent);
  }
  return out;
}

Tensor XPUNativeFunctions::pow(const Scalar& self, const Tensor& exponent) {
  Tensor out;
  auto iter = TensorIterator::binary_op(
      out, native::wrapped_scalar_tensor(self), exponent);
  native::xpu::pow_tensor_tensor_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::pow_out(
    const Scalar& self,
    const Tensor& exponent,
    Tensor& out) {
  if (self.equal(1.0)) {
    out.fill_(1);
  } else {
    return XPUNativeFunctions::pow_out(
        native::wrapped_scalar_tensor(self), exponent, out);
  }
  return out;
}

} // namespace at
