#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <aten/sycl/PowKernels.h>

namespace at {

TensorIterator pow_tensor_tensor_meta(
    const Tensor& base,
    const Tensor& exp,
    Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_binary_op(out, base, exp);
  return iter;
}

TensorIterator pow_tensor_scalar_meta(
    const Tensor& base,
    const Scalar& exp,
    Tensor& out) {
  // Numpy compatibility check:
  TORCH_CHECK(
      !(isIntegralType(base.scalar_type(), true) && exp.isIntegral(true) &&
        exp.toLong() < 0),
      "Integers to negative integer powers are not allowed.");

  auto common_dtype = at::result_type(base, exp);
  TensorIterator iter;
  iter.build_output_borrowing_argument_owning_unary_op(
      out, base.to(common_dtype));
  return iter;
}

Tensor XPUNativeFunctions::pow(const Tensor& self, const Tensor& exponent) {
  Tensor out;
  auto iter = pow_tensor_tensor_meta(self, exponent, out);
  native::xpu::pow_tensor_tensor_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::pow_(Tensor& self, const Tensor& exponent) {
  auto iter = pow_tensor_tensor_meta(self, exponent, self);
  native::xpu::pow_tensor_tensor_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::pow_out(
    const Tensor& base,
    const Tensor& exp,
    Tensor& out) {
  auto iter = pow_tensor_tensor_meta(base, exp, out);
  native::xpu::pow_tensor_tensor_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::pow(const Tensor& self, const Scalar& exponent) {
  Tensor out;
  auto iter = pow_tensor_scalar_meta(self, exponent, out);
  native::xpu::pow_tensor_scalar_kernel(iter, exponent);
  return iter.output();
}

Tensor& XPUNativeFunctions::pow_(Tensor& self, const Scalar& exponent) {
  auto iter = pow_tensor_scalar_meta(self, exponent, self);
  if (exponent.equal(0.0) || exponent.equal(false)) {
    self.fill_(1);
  } else if (exponent.equal(1.0) || exponent.equal(true)) {
  } else {
    native::xpu::pow_tensor_scalar_kernel(iter, exponent);
  }
  return self;
}

Tensor& XPUNativeFunctions::pow_out(
    const Tensor& self,
    const Scalar& exponent,
    Tensor& out) {
  auto iter = pow_tensor_scalar_meta(self, exponent, out);
  if (exponent.equal(0.0) || exponent.equal(false)) {
    out.fill_(1);
  } else if (exponent.equal(1.0) || exponent.equal(true)) {
    out.copy_(self);
  } else {
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
