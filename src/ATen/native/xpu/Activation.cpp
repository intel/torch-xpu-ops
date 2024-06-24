#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/ActivationEluKernels.h>
#include <ATen/native/xpu/sycl/ActivationGeluKernel.h>
#include <ATen/native/xpu/sycl/ActivationHardswishKernels.h>
#include <ATen/native/xpu/sycl/ActivationHardtanhKernels.h>
#include <ATen/native/xpu/sycl/ActivationSiluKernels.h>
#include <ATen/native/xpu/sycl/ActivationThresholdKernel.h>

namespace at {

Tensor XPUNativeFunctions::relu(const Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
  return at::clamp_min(self, 0);
}

Tensor& XPUNativeFunctions::relu_(Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
  return at::clamp_min_(self, 0);
}

Tensor& XPUNativeFunctions::relu_out(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
  return at::clamp_min_out(out, self, 0);
}

TensorIterator threshold_meta(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value,
    Tensor& out) {
  TensorIterator iter;
  iter.build(TensorIteratorConfig()
                 .set_check_mem_overlap(
                     false) // threshold is idempotent, so overlap is okay
                 .add_output(out)
                 .add_const_input(self)
                 .add_const_input(self) // other
                 .allow_cpu_scalars(true)
                 .promote_inputs_to_common_dtype(true)
                 .cast_common_dtype_to_outputs(true)
                 .enforce_safe_casting_to_output(true));
  return iter;
}

Tensor XPUNativeFunctions::threshold(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value) {
  Tensor out;
  auto iter = threshold_meta(self, threshold, value, out);
  native::xpu::threshold_kernel(iter, threshold, value);
  return iter.output();
}

Tensor& XPUNativeFunctions::threshold_(
    Tensor& self,
    const Scalar& threshold,
    const Scalar& value) {
  auto iter = threshold_meta(self, threshold, value, self);
  native::xpu::threshold_kernel(iter, threshold, value);
  return self;
}

Tensor& XPUNativeFunctions::threshold_out(
    const Tensor& self,
    const Scalar& threshold,
    const Scalar& value,
    Tensor& out) {
  auto iter = threshold_meta(self, threshold, value, out);
  native::xpu::threshold_kernel(iter, threshold, value);
  return out;
}

TensorIterator threshold_backward_meta(
    const Tensor& grad,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& gradInput) {
  TensorIterator iter;
  iter.build(TensorIteratorConfig()
                 .set_check_mem_overlap(
                     false) // threshold is idempotent, so overlap is okay
                 .add_output(gradInput)
                 .add_input(self)
                 .add_input(grad) // other
                 .allow_cpu_scalars(true)
                 .promote_inputs_to_common_dtype(true)
                 .cast_common_dtype_to_outputs(true)
                 .enforce_safe_casting_to_output(true));
  return iter;
}

Tensor XPUNativeFunctions::threshold_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold) {
  Tensor grad_input;
  auto iter = threshold_backward_meta(grad_output, self, threshold, grad_input);
  native::xpu::threshold_kernel(iter, threshold, 0);
  return iter.output();
}

Tensor& XPUNativeFunctions::threshold_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
  auto iter = threshold_backward_meta(grad_output, self, threshold, grad_input);
  native::xpu::threshold_kernel(iter, threshold, 0);
  return grad_input;
}

Tensor XPUNativeFunctions::gelu(
    const Tensor& self,
    c10::string_view approximate) {
  Tensor out;
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::gelu_kernel(iter, approximate);
  return iter.output();
}

Tensor& XPUNativeFunctions::gelu_(Tensor& self, c10::string_view approximate) {
  auto iter = TensorIterator::unary_op(self, self);
  native::xpu::gelu_kernel(iter, approximate);
  return self;
}

Tensor& XPUNativeFunctions::gelu_out(
    const Tensor& self,
    c10::string_view approximate,
    Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::gelu_kernel(iter, approximate);
  return out;
}

Tensor XPUNativeFunctions::gelu_backward(
    const Tensor& grad_output,
    const Tensor& self,
    c10::string_view approximate) {
  Tensor grad_input;
  auto iter =
      TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  native::xpu::gelu_backward_kernel(iter, approximate);
  return iter.output();
}

Tensor& XPUNativeFunctions::gelu_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    c10::string_view approximate,
    Tensor& grad_input) {
  auto iter =
      TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  native::xpu::gelu_backward_kernel(iter, approximate);
  return grad_input;
}

TensorIterator elu_meta(
    const Tensor& self,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    Tensor& out) {
  TensorIterator iter;
  iter = TensorIterator::unary_op(out, self);
  return iter;
}

Tensor& XPUNativeFunctions::elu_out(
    const Tensor& self,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    Tensor& out) {
  auto iter = elu_meta(self, alpha, scale, input_scale, out);
  native::xpu::elu_kernel(iter, alpha, scale, input_scale);
  return out;
}

Tensor XPUNativeFunctions::elu(
    const Tensor& self,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale) {
  Tensor out;
  auto iter = elu_meta(self, alpha, scale, input_scale, out);
  native::xpu::elu_kernel(iter, alpha, scale, input_scale);
  return iter.output();
}

Tensor& XPUNativeFunctions::elu_(
    Tensor& self,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale) {
  auto iter = elu_meta(self, alpha, scale, input_scale, self);
  native::xpu::elu_kernel(iter, alpha, scale, input_scale);
  return self;
}

TensorIterator elu_backward_meta(
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result,
    const Tensor& self_or_result,
    Tensor& grad_input) {
  TORCH_CHECK(
      !is_result || alpha.to<double>() >= 0.0,
      "In-place elu backward calculation is triggered with a negative slope which is not supported. "
      "This is caused by calling in-place forward function with a negative slope, "
      "please call out-of-place version instead.");

  TensorIterator iter;
  iter = TensorIterator::borrowing_binary_op(
      grad_input, grad_output, self_or_result);
  return iter;
}

Tensor& XPUNativeFunctions::elu_backward_out(
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result,
    const Tensor& self_or_result,
    Tensor& grad_input) {
  auto iter = elu_backward_meta(
      grad_output,
      alpha,
      scale,
      input_scale,
      is_result,
      self_or_result,
      grad_input);
  native::xpu::elu_backward_kernel(iter, alpha, scale, input_scale, is_result);
  return grad_input;
}

Tensor XPUNativeFunctions::elu_backward(
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result,
    const Tensor& self_or_result) {
  Tensor grad_input;
  auto iter = elu_backward_meta(
      grad_output,
      alpha,
      scale,
      input_scale,
      is_result,
      self_or_result,
      grad_input);
  native::xpu::elu_backward_kernel(iter, alpha, scale, input_scale, is_result);
  return iter.output();
}

Tensor XPUNativeFunctions::silu(const Tensor& self) {
  Tensor out;
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::silu_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::silu_(Tensor& self) {
  auto iter = TensorIterator::unary_op(self, self);
  native::xpu::silu_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::silu_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::silu_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::silu_backward(
    const Tensor& grad_output,
    const Tensor& self) {
  Tensor grad_input;
  auto iter =
      TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  native::xpu::silu_backward_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::silu_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    Tensor& grad_input) {
  auto iter =
      TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  native::xpu::silu_backward_kernel(iter);
  return grad_input;
}

Tensor XPUNativeFunctions::hardtanh(
    const Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  Tensor result = at::empty_like(self);
  return at::hardtanh_out(result, self, min, max);
}

Tensor& XPUNativeFunctions::hardtanh_out(
    const Tensor& self,
    const Scalar& min,
    const Scalar& max,
    Tensor& result) {
  TORCH_CHECK(
      self.scalar_type() != at::kBool,
      "Boolean inputs not supported for hardtanh");
  Scalar min_, max_;
  if (at::isIntegralType(self.scalar_type(), /*include_bool*/ false)) {
    int64_t minval = min.toLong();
    int64_t maxval = max.toLong();
    TORCH_CHECK(
        self.dtype() != at::kByte || (minval >= 0 && maxval >= 0),
        "cannot do hardtanh on an unsigned type with negative limits");
    min_ = minval;
    max_ = maxval;
  } else {
    min_ = min;
    max_ = max;
  }
  return at::clamp_out(result, self, min_, max_);
}

Tensor& XPUNativeFunctions::hardtanh_(
    Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  return at::hardtanh_out(self, self, min, max);
}

Tensor& XPUNativeFunctions::hardtanh_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& min,
    const Scalar& max,
    Tensor& grad_input) {
  auto iter =
      TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  native::xpu::hardtanh_backward_kernel(iter, min, max);
  return grad_input;
}

Tensor XPUNativeFunctions::hardtanh_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  Tensor result;
  auto iter = TensorIterator::borrowing_binary_op(result, grad_output, self);
  native::xpu::hardtanh_backward_kernel(iter, min, max);
  return iter.output();
}

Tensor XPUNativeFunctions::hardswish(const Tensor& self) {
  Tensor result;
  auto iter = TensorIterator::unary_op(result, self);
  native::xpu::hardswish_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::hardswish_out(const Tensor& self, Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  native::xpu::hardswish_kernel(iter);
  return result;
}

Tensor& XPUNativeFunctions::hardswish_(Tensor& self) {
  auto iter = TensorIterator::unary_op(self, self);
  native::xpu::hardswish_kernel(iter);
  return self;
}

Tensor XPUNativeFunctions::hardswish_backward(
    const Tensor& grad_output,
    const Tensor& self) {
  Tensor grad_input;
  auto iter =
      TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  native::xpu::hardswish_backward_kernel(iter);
  return iter.output();
}

} // namespace at
