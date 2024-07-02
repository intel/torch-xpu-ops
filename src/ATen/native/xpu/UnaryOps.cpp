#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/AbsKernel.h>
#include <ATen/native/xpu/sycl/UnaryFractionKernels.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAcosKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAcoshKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAsinKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAsinhKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAtanKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAtanhKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricCosKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricCoshKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricSinKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricTanhKernel.h>
#include <ATen/native/xpu/sycl/UnaryKernels.h>
#include <ATen/native/xpu/sycl/UnaryLogKernels.h>
#include <ATen/native/xpu/sycl/UnarySignKernels.h>
#include <ATen/native/xpu/sycl/UnarySpecialOpsKernels.h>

namespace at {

template <typename Stub>
static inline Tensor& unary_op_impl_out(
    Tensor& result,
    const Tensor& self,
    Stub& stub) {
  auto iter = TensorIterator::unary_op(result, self);
  stub(iter);
  return result;
}

template <typename Stub, typename... Args>
static inline Tensor& unary_op_impl_float_out(
    Tensor& result,
    const Tensor& self,
    Stub& stub,
    Args... args) {
  auto iter = TensorIterator::unary_float_op(result, self);
  stub(iter, args...);
  iter.cast_outputs();
  return result;
}

template <typename Stub>
static inline Tensor& unary_op_impl_with_complex_to_float_out(
    Tensor& result,
    const Tensor& self,
    Stub& stub,
    bool promotes_integer_to_float) {
  if (self.is_complex() && !result.is_complex()) {
    // Checks if the corresponding float type can be cast to the desired dtype
    const auto float_type = c10::toRealValueType(self.scalar_type());
    TORCH_CHECK(
        canCast(float_type, result.scalar_type()),
        "result type ",
        float_type,
        " can't be cast to the desired output type ",
        result.scalar_type());

    // Runs the function complex->complex, as TensorIterator expects
    Tensor complex_result = at::empty({0}, self.options());
    auto iter = TensorIterator::unary_op(complex_result, self);
    stub(iter);

    // Copies the complex result to the actual result and returns it
    at::native::resize_output(result, complex_result.sizes());
    result.copy_(at::real(complex_result));
    return result;
  }

  if (promotes_integer_to_float) {
    return unary_op_impl_float_out(result, self, stub);
  }

  return unary_op_impl_out(result, self, stub);
}

// out_impl passed into unary_op_impl and unary_op_impl_  must go through at::
// device dispatch otherwise it won't dispatch to out-of-source devices like
// XLA. For example it must be at::bitwise_not_out instead of
// bitwise_not_out(which is at::native!).
template <typename OutImpl>
static inline Tensor unary_op_impl(const Tensor& self, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

// An alternate version of unary_op_impl that follows the same pattern
// for non-complex inputs, but returns a floating point tensor
// for complex inputs by default.
template <typename OutImpl>
static inline Tensor unary_op_impl_with_complex_to_float(
    const Tensor& self,
    OutImpl& out_impl) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty_like(self, self.options().dtype(float_type));
    return out_impl(result, self);
  }

  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

template <typename OutImpl>
static inline Tensor& unary_op_impl_(Tensor& self, OutImpl& out_impl) {
  return out_impl(self, self);
}

Tensor XPUNativeFunctions::abs(const Tensor& self) {
  return unary_op_impl_with_complex_to_float(self, at::abs_out);
}

Tensor& XPUNativeFunctions::abs_(Tensor& self) {
  TORCH_CHECK(
      !self.is_complex(), "In-place abs is not supported for complex tensors.");
  return unary_op_impl_(self, at::abs_out);
}

Tensor& XPUNativeFunctions::abs_out(const Tensor& self, Tensor& out) {
  return unary_op_impl_with_complex_to_float_out(
      out,
      self,
      native::xpu::abs_kernel,
      /*promotes_integer_to_float=*/false);
}

Tensor XPUNativeFunctions::sin(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::sin_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::sin_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::sin_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::sin_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::sin_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::cos(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::cos_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::cos_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::cos_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::cos_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::cos_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::log(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::log_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::log_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::log_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::log_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::log_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::sqrt(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::sqrt_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::sqrt_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::sqrt_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::sqrt_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::sqrt_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::rsqrt(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::rsqrt_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::rsqrt_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::rsqrt_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::rsqrt_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::rsqrt_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::tanh(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::tanh_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::tanh_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::tanh_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::tanh_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::tanh_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::neg(const Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() != kBool,
      "Negation, the `-` operator, on a bool tensor is not supported. "
      "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  native::xpu::neg_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::neg_(Tensor& self) {
  TORCH_CHECK(
      self.scalar_type() != kBool,
      "Negation, the `-` operator, on a bool tensor is not supported. "
      "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  TensorIterator iter;
  iter.build_borrowing_unary_op(self, self);
  native::xpu::neg_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::neg_out(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      self.scalar_type() != kBool,
      "Negation, the `-` operator, on a bool tensor is not supported. "
      "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  native::xpu::neg_kernel(iter);
  return out;
}

TensorIterator logical_not_meta(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build(TensorIteratorConfig()
                 .check_all_same_dtype(false)
                 .add_output(out)
                 .add_const_input(self));
  return iter;
}

Tensor XPUNativeFunctions::logical_not(const Tensor& self) {
  Tensor out = at::empty({0}, self.options().dtype(kBool));
  return at::logical_not_out(out, self);
}

Tensor& XPUNativeFunctions::logical_not_(Tensor& self) {
  return at::logical_not_out(self, self);
}

Tensor& XPUNativeFunctions::logical_not_out(const Tensor& self, Tensor& out) {
  auto iter = logical_not_meta(self, out);
  native::xpu::logical_not_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::reciprocal(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::reciprocal_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::reciprocal_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::reciprocal_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::reciprocal_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::reciprocal_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::bitwise_not_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  native::xpu::bitwise_not_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::exp(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::exp_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::exp_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::exp_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::exp_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::exp_kernel(iter);
  return self;
}

Tensor XPUNativeFunctions::sigmoid(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::sigmoid_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::sigmoid_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::sigmoid_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::sigmoid_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::sigmoid_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::sgn(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  if (self.is_complex()) {
    native::xpu::sgn_kernel(iter);
  } else {
    native::xpu::sign_kernel(iter);
  }
  return iter.output();
}

Tensor& XPUNativeFunctions::sgn_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_op(self, self);
  if (self.is_complex()) {
    native::xpu::sgn_kernel(iter);
  } else {
    native::xpu::sign_kernel(iter);
  }
  return self;
}

Tensor& XPUNativeFunctions::sgn_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  if (self.is_complex()) {
    native::xpu::sgn_kernel(iter);
  } else {
    native::xpu::sign_kernel(iter);
  }
  return out;
}

Tensor XPUNativeFunctions::acos(const Tensor& self) {
  Tensor out;
  TensorIterator iter;

  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::acos_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::acos_(Tensor& self) {
  TensorIterator iter;

  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::acos_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::acos_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;

  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::acos_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::acosh(const Tensor& self) {
  Tensor out;
  TensorIterator iter;

  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::acosh_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::acosh_(Tensor& self) {
  TensorIterator iter;

  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::acosh_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::acosh_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;

  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::acosh_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::erf(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::erf_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::erf_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::erf_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::erf_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::erf_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::erfc(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::erfc_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::erfc_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::erfc_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::erfc_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::erfc_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::asinh(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::asinh_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::asinh_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::asinh_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::asinh_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::asinh_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::asin(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::asin_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::asin_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::asin_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::asin_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::asin_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::atan(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::atan_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::atan_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::atan_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::atan_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::atan_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::atanh(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::atanh_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::atanh_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::atanh_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::atanh_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::atanh_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::cosh(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::cosh_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::cosh_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::cosh_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::cosh_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::cosh_kernel(iter);
  return out;
}

} // namespace at
