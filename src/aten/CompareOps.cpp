#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/ScalarOps.h>
#include <torch/library.h>

#include <aten/sycl/CompareKernels.h>

namespace at {
namespace native {
namespace xpu {

Tensor eq_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self_arg, other_arg);
  native::xpu::eq_kernel(iter);
  return iter.output();
}

Tensor& eq_tensor_(Tensor& self, const Tensor& other_arg) {
  auto iter = TensorIterator::comparison_op(self, self, other_arg);
  native::xpu::eq_kernel(iter);
  return self;
}

Tensor eq_scalar(const Tensor& self_arg, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::eq_tensor(self_arg, wrapper);
}

Tensor& eq_scalar_(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::eq_tensor_(self, wrapper);
}

Tensor ne_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self_arg, other_arg);
  native::xpu::ne_kernel(iter);
  return iter.output();
}

Tensor& ne_tensor_(Tensor& self, const Tensor& other_arg) {
  auto iter = TensorIterator::comparison_op(self, self, other_arg);
  native::xpu::ne_kernel(iter);
  return self;
}

Tensor ne_scalar(const Tensor& self_arg, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::ne_tensor(self_arg, wrapper);
}

Tensor& ne_scalar_(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::ne_tensor_(self, wrapper);
}

Tensor lt_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self_arg, other_arg);
  native::xpu::lt_kernel(iter);
  return iter.output();
}

Tensor& lt_tensor_(Tensor& self, const Tensor& other_arg) {
  auto iter = TensorIterator::comparison_op(self, self, other_arg);
  native::xpu::lt_kernel(iter);
  return self;
}

Tensor lt_scalar(const Tensor& self_arg, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::lt_tensor(self_arg, wrapper);
}

Tensor& lt_scalar_(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::lt_tensor_(self, wrapper);
}

Tensor le_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self_arg, other_arg);
  native::xpu::le_kernel(iter);
  return iter.output();
}

Tensor& le_tensor_(Tensor& self, const Tensor& other_arg) {
  auto iter = TensorIterator::comparison_op(self, self, other_arg);
  native::xpu::le_kernel(iter);
  return self;
}

Tensor le_scalar(const Tensor& self_arg, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::le_tensor(self_arg, wrapper);
}

Tensor& le_scalar_(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::le_tensor_(self, wrapper);
}

Tensor gt_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self_arg, other_arg);
  native::xpu::gt_kernel(iter);
  return iter.output();
}

Tensor& gt_tensor_(Tensor& self, const Tensor& other_arg) {
  auto iter = TensorIterator::comparison_op(self, self, other_arg);
  native::xpu::gt_kernel(iter);
  return self;
}

Tensor gt_scalar(const Tensor& self_arg, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::gt_tensor(self_arg, wrapper);
}

Tensor& gt_scalar_(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::gt_tensor_(self, wrapper);
}

Tensor ge_tensor(const Tensor& self_arg, const Tensor& other_arg) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self_arg, other_arg);
  native::xpu::ge_kernel(iter);
  return iter.output();
}

Tensor& ge_tensor_(Tensor& self, const Tensor& other_arg) {
  auto iter = TensorIterator::comparison_op(self, self, other_arg);
  native::xpu::ge_kernel(iter);
  return self;
}

Tensor ge_scalar(const Tensor& self_arg, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::ge_tensor(self_arg, wrapper);
}

Tensor& ge_scalar_(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  return native::xpu::ge_tensor_(self, wrapper);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::eq.Scalar"), TORCH_FN(eq_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::eq_.Scalar"), TORCH_FN(eq_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::eq.Tensor"), TORCH_FN(eq_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::eq_.Tensor"), TORCH_FN(eq_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::ne.Scalar"), TORCH_FN(ne_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::ne_.Scalar"), TORCH_FN(ne_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::ne.Tensor"), TORCH_FN(ne_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::ne_.Tensor"), TORCH_FN(ne_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::lt.Scalar"), TORCH_FN(lt_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::lt_.Scalar"), TORCH_FN(lt_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::lt.Tensor"), TORCH_FN(lt_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::lt_.Tensor"), TORCH_FN(lt_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::le.Scalar"), TORCH_FN(le_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::le_.Scalar"), TORCH_FN(le_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::le.Tensor"), TORCH_FN(le_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::le_.Tensor"), TORCH_FN(le_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::gt.Scalar"), TORCH_FN(gt_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::gt_.Scalar"), TORCH_FN(gt_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::gt.Tensor"), TORCH_FN(gt_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::gt_.Tensor"), TORCH_FN(gt_tensor_));
  m.impl(TORCH_SELECTIVE_NAME("aten::ge.Scalar"), TORCH_FN(ge_scalar));
  m.impl(TORCH_SELECTIVE_NAME("aten::ge_.Scalar"), TORCH_FN(ge_scalar_));
  m.impl(TORCH_SELECTIVE_NAME("aten::ge.Tensor"), TORCH_FN(ge_tensor));
  m.impl(TORCH_SELECTIVE_NAME("aten::ge_.Tensor"), TORCH_FN(ge_tensor_));
}

}}} // at::native::xpu
