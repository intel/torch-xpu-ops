#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <aten/sycl/UnaryKernels.h>
#include <aten/sycl/UnaryLogKernels.h>

namespace at {

Tensor XPUNativeFunctions::abs(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::abs_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::abs_(Tensor& self) {
  auto iter = TensorIterator::unary_op(self, self);
  native::xpu::abs_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::abs_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::abs_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::sin(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::sin_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::sin_(Tensor& self) {
  auto iter = TensorIterator::unary_float_op(self, self);
  native::xpu::sin_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::sin_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::sin_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::cos(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::cos_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::cos_(Tensor& self) {
  auto iter = TensorIterator::unary_float_op(self, self);
  native::xpu::cos_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::cos_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::cos_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::log(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::log_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::log_(Tensor& self) {
  auto iter = TensorIterator::unary_float_op(self, self);
  native::xpu::log_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::log_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::log_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::sqrt(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::sqrt_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::sqrt_(Tensor& self) {
  auto iter = TensorIterator::unary_float_op(self, self);
  native::xpu::sqrt_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::sqrt_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::sqrt_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::rsqrt(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::rsqrt_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::rsqrt_(Tensor& self) {
  auto iter = TensorIterator::unary_float_op(self, self);
  native::xpu::rsqrt_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::rsqrt_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::rsqrt_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::tanh(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::tanh_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::tanh_(Tensor& self) {
  auto iter = TensorIterator::unary_float_op(self, self);
  native::xpu::tanh_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::tanh_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_float_op(out, self);
  native::xpu::tanh_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::neg(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::neg_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::neg_(Tensor& self) {
  auto iter = TensorIterator::unary_op(self, self);
  native::xpu::neg_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::neg_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::neg_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::reciprocal(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::reciprocal_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::reciprocal_(Tensor& self) {
  auto iter = TensorIterator::unary_op(self, self);
  native::xpu::reciprocal_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::reciprocal_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::reciprocal_kernel(iter);
  return out;
}

} // namespace at
