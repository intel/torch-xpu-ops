#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/LinearAlgebraKernels.h>

namespace at {

static void check_1d(const Tensor& t, const char* arg, const char* fn) {
  TORCH_CHECK(
      t.dim() == 1,
      fn,
      ": Expected 1-D argument ",
      arg,
      ", but got ",
      t.dim(),
      "-D");
}

static void check_addr_scalar(
    const ScalarType dtype,
    const Scalar& scalar,
    const std::string& scalar_name) {
  TORCH_CHECK(
      !scalar.isBoolean() || dtype == ScalarType::Bool,
      "Boolean ",
      scalar_name,
      " only supported for Boolean results.");
  TORCH_CHECK(
      isFloatingType(dtype) || isComplexType(dtype) || scalar.isIntegral(true),
      "For integral input tensors, "
      "argument ",
      scalar_name,
      " must not be a floating point number.");
}

static TensorIterator build_addr_iter(
    Tensor& result,
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2) {
  check_1d(vec1, "vec1", "addr");
  check_1d(vec2, "vec2", "addr");

  const auto vec1_size0 = vec1.sizes()[0];
  const auto vec2_size0 = vec2.sizes()[0];
  auto self_ = &result == &self
      ? c10::MaybeOwned<Tensor>::borrowed(self)
      : expand_size(self, {vec1_size0, vec2_size0}, "addr");
  TORCH_CHECK(
      self_->dim() == 2,
      "2D tensor expected, got ",
      self_->dim(),
      "D tensor for input");
  TORCH_CHECK(
      self_->sizes()[0] == vec1_size0 && self_->sizes()[1] == vec2_size0,
      "size mismatch, input: ",
      self_->sizes(),
      ", v1: ",
      vec1.sizes(),
      ", v2: ",
      vec2.sizes());

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(result)
                  .add_owned_const_input(*self_)
                  .add_owned_const_input(vec1.reshape({vec1_size0, 1}))
                  .add_const_input(vec2)
                  .allow_cpu_scalars(true)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .build();
  return iter;
}

Tensor XPUNativeFunctions::addr(
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    const Scalar& beta,
    const Scalar& alpha) {
  Tensor result;
  auto iter = build_addr_iter(result, self, vec1, vec2);

  check_addr_scalar(iter.dtype(), beta, "beta");
  check_addr_scalar(iter.dtype(), alpha, "alpha");

  native::xpu::addr_kernel(iter, beta, alpha);
  return iter.output();
}

Tensor& XPUNativeFunctions::addr_out(
    const Tensor& self,
    const Tensor& vec1,
    const Tensor& vec2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  auto iter = build_addr_iter(out, self, vec1, vec2);
  check_addr_scalar(iter.dtype(), beta, "beta");
  check_addr_scalar(iter.dtype(), alpha, "alpha");

  native::xpu::addr_kernel(iter, beta, alpha);
  return out;
}
} // namespace at
