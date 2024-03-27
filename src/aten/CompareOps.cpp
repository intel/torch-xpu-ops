#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/CompareKernels.h>

namespace at {

Tensor XPUNativeFunctions::eq(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::eq_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::eq_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::eq_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::eq_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::eq_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::eq(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::eq(self, wrapper);
}

Tensor& XPUNativeFunctions::eq_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::eq_(self, wrapper);
}

Tensor& XPUNativeFunctions::eq_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::eq_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::ne(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::ne_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::ne_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::ne_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::ne_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::ne_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::ne(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ne(self, wrapper);
}

Tensor& XPUNativeFunctions::ne_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ne_(self, wrapper);
}

Tensor& XPUNativeFunctions::ne_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ne_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::lt(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::lt_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::lt_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::lt_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::lt_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::lt_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::lt(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::lt(self, wrapper);
}

Tensor& XPUNativeFunctions::lt_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::lt_(self, wrapper);
}

Tensor& XPUNativeFunctions::lt_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::lt_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::le(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::le_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::le_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::le_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::le_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::le_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::le(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::le(self, wrapper);
}

Tensor& XPUNativeFunctions::le_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::le_(self, wrapper);
}

Tensor& XPUNativeFunctions::le_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::le_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::gt(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::gt_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::gt_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::gt_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::gt_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::gt_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::gt(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::gt(self, wrapper);
}

Tensor& XPUNativeFunctions::gt_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::gt_(self, wrapper);
}

Tensor& XPUNativeFunctions::gt_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::gt_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::ge(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::ge_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::ge_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::ge_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::ge_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::ge_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::ge(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ge(self, wrapper);
}

Tensor& XPUNativeFunctions::ge_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ge_(self, wrapper);
}

Tensor& XPUNativeFunctions::ge_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ge_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::isnan(const Tensor& self) {
  return XPUNativeFunctions::ne(self, self);
}

Tensor& XPUNativeFunctions::isnan_out(const Tensor& self, Tensor& out) {
  return XPUNativeFunctions::ne_out(self, self, out);
}

} // namespace at
