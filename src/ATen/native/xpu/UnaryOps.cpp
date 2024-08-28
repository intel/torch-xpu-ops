#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/AbsKernel.h>
#include <ATen/native/xpu/sycl/UnaryComplexKernels.h>
#include <ATen/native/xpu/sycl/UnaryFractionKernels.h>
#include <ATen/native/xpu/sycl/UnaryGammaKernels.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAcosKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAcoshKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAsinKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAsinhKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAtanKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricAtanhKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricCosKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricCoshKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricSinKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricSinhKernel.h>
#include <ATen/native/xpu/sycl/UnaryGeometricTanKernel.h>
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

Tensor XPUNativeFunctions::digamma(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::digamma_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::digamma_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::digamma_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::digamma_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::digamma_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::polygamma(int64_t n, const Tensor& self) {
  TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::polygamma_kernel(iter, n);
  return iter.output();
}

Tensor& XPUNativeFunctions::polygamma_out(
    int64_t n,
    const Tensor& self,
    Tensor& out) {
  TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::polygamma_kernel(iter, n);
  return out;
}

Tensor& XPUNativeFunctions::polygamma_(Tensor& self, int64_t n) {
  return polygamma_out(n, self, self);
}

Tensor XPUNativeFunctions::lgamma(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::lgamma_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::lgamma_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::lgamma_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::lgamma_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::lgamma_kernel(iter);
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

Tensor XPUNativeFunctions::log10(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::log10_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::log10_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::log10_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::log10_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::log10_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::log1p(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::log1p_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::log1p_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::log1p_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::log1p_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::log1p_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::log2(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::log2_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::log2_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::log2_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::log2_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::log2_kernel(iter);
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

Tensor XPUNativeFunctions::sign(const Tensor& self) {
  TORCH_CHECK(
      !self.is_complex(),
      "Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.");
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  native::xpu::sign_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::sign_(Tensor& self) {
  TORCH_CHECK(
      !self.is_complex(),
      "Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.");
  TensorIterator iter;
  iter.build_borrowing_unary_op(self, self);
  native::xpu::sign_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::sign_out(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      !self.is_complex(),
      "Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.");
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  native::xpu::sign_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::signbit(const Tensor& self) {
  TORCH_CHECK(
      !self.is_complex(), "signbit is not implemented for complex tensors.");

  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_force_boolean_op(out, self);

  if (self.dtype() == at::kBool) {
    iter.output().fill_(false);
  } else {
    native::xpu::signbit_kernel(iter);
  }
  return iter.output();
}

Tensor& XPUNativeFunctions::signbit_out(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      !self.is_complex(), "signbit is not implemented for complex tensors.");
  TORCH_CHECK(
      out.dtype() == at::kBool,
      "signbit does not support non-boolean outputs.");

  TensorIterator iter;
  iter.build_borrowing_unary_force_boolean_op(out, self);

  if (self.dtype() == at::kBool) {
    out.fill_(false);
  } else {
    native::xpu::signbit_kernel(iter);
  }
  return out;
}

Tensor& XPUNativeFunctions::logit_out(
    const Tensor& self,
    std::optional<double> eps,
    Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::logit_kernel(iter, Scalar(eps ? eps.value() : -1.0));
  return out;
}

Tensor XPUNativeFunctions::logit(
    const Tensor& self,
    std::optional<double> eps) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::logit_kernel(iter, Scalar(eps ? eps.value() : -1.0));
  return iter.output();
}

Tensor& XPUNativeFunctions::logit_(Tensor& self, std::optional<double> eps) {
  return at::logit_out(self, self, eps);
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

Tensor XPUNativeFunctions::erfinv(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::erfinv_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::erfinv_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::erfinv_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::erfinv_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::erfinv_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::exp2(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::exp2_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::exp2_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::exp2_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::exp2_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::exp2_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::expm1(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::expm1_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::expm1_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::expm1_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::expm1_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::expm1_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::frac(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  native::xpu::frac_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::frac_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_op(self, self);
  native::xpu::frac_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::frac_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  native::xpu::frac_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::sinh(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::sinh_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::sinh_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::sinh_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::sinh_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::sinh_kernel(iter);
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

Tensor XPUNativeFunctions::tan(const Tensor& self) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::tan_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::tan_(Tensor& self) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(self, self);
  native::xpu::tan_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::tan_out(const Tensor& self, Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_float_op(out, self);
  native::xpu::tan_kernel(iter);
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

Tensor& XPUNativeFunctions::conj_physical_out(const Tensor& self, Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::conj_physical_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::conj_physical_(Tensor& self) {
  if (!self.is_complex())
    return self;
  return XPUNativeFunctions::conj_physical_out(self, self);
}

TensorIterator ceil_meta(const Tensor& self, Tensor& out) {
  TORCH_CHECK(!self.is_complex(), "ceil is not supported for complex inputs");
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  return iter;
}

Tensor XPUNativeFunctions::ceil(const Tensor& self) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    return self.clone();
  }
  Tensor out;
  auto iter = ceil_meta(self, out);
  native::xpu::ceil_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::ceil_(Tensor& self) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    return self;
  }
  auto iter = ceil_meta(self, self);
  native::xpu::ceil_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::ceil_out(const Tensor& self, Tensor& out) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    out.copy_(self);
    return out;
  }
  auto iter = ceil_meta(self, out);
  native::xpu::ceil_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::round(const Tensor& self) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    return self.clone();
  }
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  native::xpu::round_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::round_(Tensor& self) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    return self;
  }
  TensorIterator iter;
  iter.build_borrowing_unary_op(self, self);
  native::xpu::round_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::round_out(const Tensor& self, Tensor& out) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    out.copy_(self);
    return out;
  }
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  native::xpu::round_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::round(const Tensor& self, int64_t decimals) {
  Tensor out;
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  if (decimals != 0) {
    native::xpu::round_decimals_kernel(iter, decimals);
  } else {
    native::xpu::round_kernel(iter);
  }
  return iter.output();
}

Tensor& XPUNativeFunctions::round_(Tensor& self, int64_t decimals) {
  TensorIterator iter;
  iter.build_borrowing_unary_op(self, self);
  if (decimals != 0) {
    native::xpu::round_decimals_kernel(iter, decimals);
  } else {
    native::xpu::round_kernel(iter);
  }
  return self;
}

Tensor& XPUNativeFunctions::round_out(
    const Tensor& self,
    int64_t decimals,
    Tensor& out) {
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  if (decimals != 0) {
    native::xpu::round_decimals_kernel(iter, decimals);
  } else {
    native::xpu::round_kernel(iter);
  }
  return out;
}

TensorIterator meta_floor(const Tensor& self, Tensor& out) {
  // Note: this is consistent with NumPy
  TORCH_CHECK(!self.is_complex(), "floor is not supported for complex inputs");
  TensorIterator iter;
  iter.build_borrowing_unary_op(out, self);
  return iter;
}

Tensor XPUNativeFunctions::floor(const Tensor& self) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    return self.clone();
  }
  Tensor out;
  auto iter = meta_floor(self, out);
  native::xpu::floor_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::floor_(Tensor& self) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    return self;
  }
  auto iter = meta_floor(self, self);
  native::xpu::floor_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::floor_out(const Tensor& self, Tensor& out) {
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
    out.copy_(self);
    return out;
  }
  auto iter = meta_floor(self, out);
  native::xpu::floor_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::nan_to_num_out(
    const Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf,
    Tensor& result) {
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "nan_to_num: dtype of out: ",
      result.scalar_type(),
      " should be same as input: ",
      self.scalar_type());

  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    at::native::resize_output(result, self.sizes());
    result.copy_(self);
    return result;
  }

  auto iter = TensorIterator::unary_op(result, self);
  native::xpu::nan_to_num_kernel(iter, nan, pos_inf, neg_inf);
  return result;
}

} // namespace at
