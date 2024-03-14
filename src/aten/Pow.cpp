#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/PowKernels.h>

namespace at {

Tensor& XPUNativeFunctions::pow_out(
    const Tensor& base,
    const Tensor& exp,
    Tensor& result) {
  auto iter = TensorIterator::binary_op(result, base, exp);
  native::xpu::pow_tensor_tensor_kernel(iter);
  return result;
}

Tensor& XPUNativeFunctions::pow_out(
    const Scalar& base,
    const Tensor& exp,
    Tensor& result) {
  if (base.isComplex() && base.toComplexDouble() == 1.0) {
    result.resize_as_(exp).fill_(1);
  } else if (!base.isComplex() && base.toDouble() == 1.0) {
    result.resize_as_(exp).fill_(1);
  } else {
    XPUNativeFunctions::pow_out(
        native::wrapped_scalar_tensor(base), exp, result);
  }
  return result;
}

Tensor& XPUNativeFunctions::pow_out(
    const Tensor& self,
    const Scalar& exponent,
    Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  auto& base_ = iter.tensor(1);
  if (exponent.isComplex() && (exponent.toComplexDouble() == 0.0)) {
    result.resize_as_(base_).fill_(1);
  } else if (exponent.isComplex() && (exponent.toComplexDouble() == 1.0)) {
    result.resize_as_(base_).fill_(base_);
  } else if (!exponent.isComplex() && (exponent.toDouble() == 0.0)) {
    result.resize_as_(base_).fill_(1);
  } else if (!exponent.isComplex() && (exponent.toDouble() == 1.0)) {
    result.resize_as_(base_).copy_(base_);
  } else {
    native::xpu::pow_tensor_scalar_kernel(iter, exponent);
  }
  return result;
}

} // namespace at
