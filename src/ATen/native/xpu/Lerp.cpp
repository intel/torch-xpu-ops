#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Lerp.h>

namespace at {

TensorIterator lerp_Tensor_meta(
    const Tensor& self,
    const Tensor& end,
    const Tensor& weight,
    Tensor& out) {
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(),
              " for `end` but got dtype ", end.dtype());
  TORCH_CHECK(self.dtype() == weight.dtype(), "expected dtype ", self.dtype(),
              " for `weight` but got dtype ", weight.dtype());
  TensorIterator iter;
  iter.build(TensorIteratorConfig()
                .add_output(out)
                .add_const_input(self)
                .add_const_input(end)
                .add_const_input(weight));
  return iter;
}

Tensor XPUNativeFunctions::lerp_Tensor(
    const Tensor& self,
    const Tensor& end,
    const Tensor& weight) {
  Tensor out;
  auto iter = lerp_tensor_meta(self, end, weight, out);
  native::xpu::lerp_tensor_kernel(iter);
  return iter.output();
}

Tensor XPUNativeFunctions::lerp_Tensor_(
    const Tensor& self,
    const Tensor& end,
    const Tensor& weight) {
  auto iter = lerp_tensor_meta(self, end, weight, self);
  native::xpu::lerp_tensor_kernel(iter);
  return self;
}

Tensor XPUNativeFunctions::lerp_Tensor_out(
    const Tensor& self,
    const Tensor& end,
    const Tensor& weight
    Tensor& out) {
  auto iter = lerp_tensor_meta(self, end, weight, out);
  native::xpu::lerp_tensor_kernel(iter);
  return out;
}

TensorIterator lerp_Scalar_meta(
    const Tensor& self,
    const Tensor& end,
    const Scalar& /*weight*/,
    Tensor& out) {
  TORCH_CHECK(self.dtype() == end.dtype(), "expected dtype ", self.dtype(),
              " for `end` but got dtype ", end.dtype());
  TensorIterator iter;
  iter.build_binary_op(out, self, end);
  return iter;
}

Tensor XPUNativeFunctions::lerp_Scalar(
    const Tensor& self,
    const Tensor& end,
    const Scalar& weight) {
  Tensor out;
  auto iter = lerp_Scalar_meta(self, end, weight, out);
  native::xpu::lerp_scalar_kernel(iter, weight);
  return iter.output();
}

Tensor XPUNativeFunctions::lerp_Scalar_(
    const Tensor& self,
    const Tensor& end,
    const Scalar& weight) {
  auto iter = lerp_Scalar_meta(self, end, weight, self);
  native::xpu::lerp_scalar_kernel(iter, weight);
  return self;
}

Tensor XPUNativeFunctions::lerp_Scalar_out(
    const Tensor& self,
    const Tensor& end,
    const Scalar& /*weight*/,
    Tensor& out) {
  auto iter = lerp_Scalar_meta(self, end, weight, out);
  native::xpu::lerp_scalar_kernel(iter, weight);
  return out;
}

} // namespace at
