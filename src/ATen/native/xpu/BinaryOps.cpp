#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/BinaryBitwiseOpsKernels.h>
#include <ATen/native/xpu/sycl/BinaryGeometricKernels.h>
#include <ATen/native/xpu/sycl/BinaryKernels.h>
#include <ATen/native/xpu/sycl/BinaryLogicalOpsKernels.h>
#include <ATen/native/xpu/sycl/BinaryMiscBackwardOpsKernels.h>
#include <ATen/native/xpu/sycl/BinaryRemainderKernel.h>
#include <ATen/native/xpu/sycl/BinaryShiftOpsKernels.h>
#include <ATen/native/xpu/sycl/CopysignKernel.h>
#include <ATen/native/xpu/sycl/GcdLcmKernels.h>
#include <ATen/native/xpu/sycl/LogAddExpKernels.h>
#include <ATen/native/xpu/sycl/MaxMinElementwiseKernels.h>
#include <ATen/native/xpu/sycl/StepKernels.h>

namespace at {
Tensor XPUNativeFunctions::add(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return iter.output();
}

Tensor& XPUNativeFunctions::add_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  auto iter = TensorIterator::borrowing_binary_op(self, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return self;
}

Tensor& XPUNativeFunctions::add_out(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::add_kernel(iter, alpha);
  return out;
}

Tensor XPUNativeFunctions::sub(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  Tensor out;
  native::sub_check(self, other);
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::sub_kernel(iter, alpha);
  return iter.output();
}

Tensor& XPUNativeFunctions::sub_(
    Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  native::sub_check(self, other);
  auto iter = TensorIterator::borrowing_binary_op(self, self, other);
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
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::alpha_check(iter.dtype(), alpha);
  native::xpu::sub_kernel(iter, alpha);
  return out;
}

Tensor XPUNativeFunctions::mul(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::mul_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::mul_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::borrowing_binary_op(self, self, other);
  native::xpu::mul_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::mul_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::mul_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::div(const Tensor& self, const Tensor& other) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_binary_float_op(out, self, other);
  native::xpu::div_true_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::div_(Tensor& self, const Tensor& other) {
  TensorIterator iter;
  iter.build_borrowing_binary_float_op(self, self, other);
  native::xpu::div_true_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::div_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_binary_float_op(out, self, other);
  native::xpu::div_true_kernel(iter);
  return out;
}

static inline TensorIterator meta_func_div_Tensor_mode(
    const Tensor& self,
    const Tensor& other,
    const Tensor& output,
    c10::optional<c10::string_view> rounding_mode) {
  TensorIterator iter;
  if (!rounding_mode.has_value()) {
    iter.build_borrowing_binary_float_op(output, self, other);
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (*rounding_mode == "trunc") {
    iter.build_borrowing_binary_op(output, self, other);
  } else if (*rounding_mode == "floor") {
    iter.build_borrowing_binary_op(output, self, other);
  } else {
    TORCH_CHECK(
        false,
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        "but found '",
        *rounding_mode,
        "'");
  }
  return iter;
}

static inline void impl_func_div_Tensor_mode(
    TensorIterator& iter,
    ::std::optional<c10::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    native::xpu::div_true_kernel(iter);
  } else if (*rounding_mode == "trunc") {
    native::xpu::div_trunc_kernel(iter);
  } else if (*rounding_mode == "floor") {
    native::xpu::div_floor_kernel(iter);
  }
}

Tensor XPUNativeFunctions::div(
    const at::Tensor& self,
    const at::Tensor& other,
    ::std::optional<c10::string_view> rounding_mode) {
  Tensor output;
  TensorIterator iter =
      meta_func_div_Tensor_mode(self, other, output, rounding_mode);
  impl_func_div_Tensor_mode(iter, rounding_mode);
  return iter.output();
}

Tensor& XPUNativeFunctions::div_(
    at::Tensor& self,
    const at::Tensor& other,
    ::std::optional<c10::string_view> rounding_mode) {
  TensorIterator iter =
      meta_func_div_Tensor_mode(self, other, self, rounding_mode);
  impl_func_div_Tensor_mode(iter, rounding_mode);
  return self;
}

Tensor& XPUNativeFunctions::div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    ::std::optional<c10::string_view> rounding_mode,
    at::Tensor& output) {
  TensorIterator iter =
      meta_func_div_Tensor_mode(self, other, output, rounding_mode);
  impl_func_div_Tensor_mode(iter, rounding_mode);
  return output;
}

Tensor XPUNativeFunctions::rsub(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha) {
  return XPUNativeFunctions::sub(other, self, alpha);
}

Tensor XPUNativeFunctions::remainder(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::remainder_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::remainder_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::borrowing_binary_op(self, self, other);
  native::xpu::remainder_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::remainder_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::remainder_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::remainder(const Scalar& self, const Tensor& other) {
  auto wrapper = native::wrapped_scalar_tensor(self);
  return XPUNativeFunctions::remainder(wrapper, other);
}

Tensor XPUNativeFunctions::fmod(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::fmod_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::fmod_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::borrowing_binary_op(self, self, other);
  native::xpu::fmod_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::fmod_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::fmod_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::tanh_backward(
    const Tensor& grad_output,
    const Tensor& output) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, grad_output, output);
  native::xpu::tanh_backward_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::tanh_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  auto iter =
      TensorIterator::borrowing_binary_op(grad_input, grad_output, output);
  native::xpu::tanh_backward_kernel(iter);
  return grad_input;
}

Tensor& XPUNativeFunctions::bitwise_and_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::bitwise_and_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::bitwise_or_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::bitwise_or_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::bitwise_xor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::bitwise_xor_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::__lshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  native::xpu::lshift_kernel(iter);
  return iter.output();
}

Tensor XPUNativeFunctions::__lshift__(const Tensor& self, const Scalar& other) {
  Tensor result;
  auto wrapper = native::wrapped_scalar_tensor(other);
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  native::xpu::lshift_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::__ilshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  native::xpu::lshift_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::__ilshift__(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  native::xpu::lshift_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::bitwise_left_shift_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  auto iter = TensorIterator::borrowing_binary_op(result, self, other);
  native::xpu::lshift_kernel(iter);
  return result;
}

Tensor XPUNativeFunctions::__rshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  native::xpu::rshift_kernel(iter);
  return iter.output();
}

Tensor XPUNativeFunctions::__rshift__(const Tensor& self, const Scalar& other) {
  Tensor result;
  auto wrapper = native::wrapped_scalar_tensor(other);
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  native::xpu::rshift_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::__irshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  native::xpu::rshift_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::__irshift__(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  native::xpu::rshift_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::bitwise_right_shift_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& result) {
  auto iter = TensorIterator::borrowing_binary_op(result, self, other);
  native::xpu::rshift_kernel(iter);
  return result;
}

Tensor XPUNativeFunctions::gcd(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::gcd_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::gcd_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::borrowing_binary_op(self, self, other);
  native::xpu::gcd_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::gcd_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::gcd_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::nextafter(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::nextafter_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::nextafter_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::borrowing_binary_op(self, self, other);
  native::xpu::nextafter_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::nextafter_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::nextafter_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::hypot(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::hypot_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::hypot_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::borrowing_binary_op(self, self, other);
  native::xpu::hypot_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::hypot_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::hypot_kernel(iter);
  return out;
}

static inline TensorIterator meta_func_maximum(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  TORCH_CHECK(
      !self.is_complex() && !other.is_complex(),
      "maximum not implemented for complex tensors.");
  auto iter = TensorIterator::borrowing_binary_op(output, self, other);
  return iter;
}

Tensor XPUNativeFunctions::maximum(const Tensor& self, const Tensor& other) {
  Tensor output;
  auto iter = meta_func_maximum(self, other, output);
  native::xpu::maximum_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::maximum_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  auto iter = meta_func_maximum(self, other, output);
  native::xpu::maximum_kernel(iter);
  return output;
}

static inline TensorIterator meta_func_minimum(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  TORCH_CHECK(
      !self.is_complex() && !other.is_complex(),
      "minimum not implemented for complex tensors.");
  auto iter = TensorIterator::borrowing_binary_op(output, self, other);
  return iter;
}

Tensor XPUNativeFunctions::minimum(const Tensor& self, const Tensor& other) {
  Tensor output;
  auto iter = meta_func_minimum(self, other, output);
  native::xpu::minimum_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::minimum_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  auto iter = meta_func_minimum(self, other, output);
  native::xpu::minimum_kernel(iter);
  return output;
}

Tensor& XPUNativeFunctions::logit_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    std::optional<double> eps,
    Tensor& grad_input) {
  TensorIterator iter;
  iter.build_borrowing_binary_op(grad_input, grad_output, input);
  native::xpu::logit_backward_kernel(iter, Scalar(eps ? eps.value() : -1.0));
  return grad_input;
}

Tensor XPUNativeFunctions::logit_backward(
    const Tensor& grad_output,
    const Tensor& input,
    std::optional<double> eps) {
  Tensor grad_input;
  TensorIterator iter;
  iter.build_borrowing_binary_op(grad_input, grad_output, input);
  native::xpu::logit_backward_kernel(iter, Scalar(eps ? eps.value() : -1.0));
  return iter.output();
}

Tensor& XPUNativeFunctions::sigmoid_backward_out(
    const Tensor& grad_output,
    const Tensor& output,
    Tensor& grad_input) {
  TensorIterator iter;
  iter.build_borrowing_binary_op(grad_input, grad_output, output);
  native::xpu::sigmoid_backward_kernel(iter);
  return grad_input;
}

Tensor XPUNativeFunctions::sigmoid_backward(
    const Tensor& grad_output,
    const Tensor& output) {
  Tensor grad_input;
  TensorIterator iter;
  iter.build_borrowing_binary_op(grad_input, grad_output, output);
  native::xpu::sigmoid_backward_kernel(iter);
  return iter.output();
}

Tensor XPUNativeFunctions::logaddexp(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::logaddexp_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::logaddexp_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::logaddexp_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::logaddexp2(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::logaddexp2_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::logaddexp2_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::borrowing_binary_op(out, self, other);
  native::xpu::logaddexp2_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::floor_divide_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  auto iter = TensorIterator::binary_op(output, self, other);
  native::xpu::div_floor_kernel(iter);
  if (!output.defined()) {
    output = iter.output();
  }
  return output;
}

Tensor XPUNativeFunctions::floor_divide(
    const Tensor& self,
    const Tensor& other) {
  Tensor output;
  auto iter = TensorIterator::binary_op(output, self, other);
  native::xpu::div_floor_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::floor_divide_(Tensor& self, const Tensor& other) {
  return XPUNativeFunctions::floor_divide_out(self, other, self);
}

TensorIterator meta_fmin_fmax(
    const char* const name,
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  TORCH_CHECK(
      !self.is_complex() && !other.is_complex(),
      name,
      " not implemented for complex tensors.");
  TensorIterator iter;
  iter.build_binary_op(output, self, other);
  return iter;
}

Tensor& XPUNativeFunctions::fmax_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  auto iter = meta_fmin_fmax("fmax", self, other, output);
  native::xpu::fmax_kernel(iter);
  return output;
}

Tensor XPUNativeFunctions::fmax(const Tensor& self, const Tensor& other) {
  Tensor output;
  auto iter = meta_fmin_fmax("fmax", self, other, output);
  native::xpu::fmax_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::fmin_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& output) {
  auto iter = meta_fmin_fmax("fmin", self, other, output);
  native::xpu::fmin_kernel(iter);
  return output;
}

Tensor XPUNativeFunctions::fmin(const Tensor& self, const Tensor& other) {
  Tensor output;
  auto iter = meta_fmin_fmax("fmin", self, other, output);
  native::xpu::fmin_kernel(iter);
  return iter.output();
}

Tensor XPUNativeFunctions::atan2(const Tensor& self, const Tensor& other) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_binary_float_op(out, self, other);
  native::xpu::atan2_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::atan2_(Tensor& self, const Tensor& other) {
  TensorIterator iter;
  iter.build_borrowing_binary_float_op(self, self, other);
  native::xpu::atan2_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::atan2_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_binary_float_op(out, self, other);
  native::xpu::atan2_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::copysign_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_binary_float_op(out, self, other);
  native::xpu::copysign_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::copysign_(Tensor& self, const Tensor& other) {
  return XPUNativeFunctions::copysign_out(self, other, self);
}

Tensor XPUNativeFunctions::copysign(const Tensor& self, const Tensor& other) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_binary_float_op(out, self, other);
  native::xpu::copysign_kernel(iter);
  return iter.output();
}

// We need explicit cast to OutFunc because each *_out func is overloaded twice.
// Without An explicit cast, merely referring to *_out function is ambiguous.
using OutFunc =
    std::add_const<Tensor& (&)(Tensor&, const Tensor&, const Tensor&)>::type;

template <typename OutImpl>
Tensor comparison_op(
    const Tensor& self,
    const Tensor& other,
    OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return out_impl(result, self, other);
}

template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Tensor& other, OutImpl& out_impl) {
  return out_impl(self, self, other);
}

template <typename OutImpl>
Tensor& comparison_op_out(
    Tensor& result,
    const Tensor& self,
    const Scalar& other,
    OutImpl& out_impl) {
  return out_impl(result, self, native::wrapped_scalar_tensor(other));
}

template <typename OutImpl>
Tensor comparison_op(
    const Tensor& self,
    const Scalar& other,
    OutImpl& out_impl) {
  return comparison_op(self, native::wrapped_scalar_tensor(other), out_impl);
}

template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return out_impl(self, self, native::wrapped_scalar_tensor(other));
}

Tensor& XPUNativeFunctions::logical_and_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::logical_and_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::logical_and(
    const Tensor& self,
    const Tensor& other) {
  return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out));
}

Tensor& XPUNativeFunctions::logical_and_(Tensor& self, const Tensor& other) {
  return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out));
}

Tensor& XPUNativeFunctions::logical_or_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::logical_or_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::logical_or(const Tensor& self, const Tensor& other) {
  return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out));
}

Tensor& XPUNativeFunctions::logical_or_(Tensor& self, const Tensor& other) {
  return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out));
}

Tensor& XPUNativeFunctions::logical_xor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::logical_xor_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::logical_xor(
    const Tensor& self,
    const Tensor& other) {
  return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out));
}

Tensor& XPUNativeFunctions::logical_xor_(Tensor& self, const Tensor& other) {
  return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out));
}

} // namespace at
