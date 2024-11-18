
#include <comm/xpu_aten.h>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/CopyKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/UnaryKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SqrtFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::sqrt(a);
  }
};

template <typename scalar_t>
struct RsqrtFunctor {
  scalar_t operator()(scalar_t a) const {
    return sycl::rsqrt(float(a));
  }
};

template <>
struct RsqrtFunctor<double> {
  double operator()(double a) const {
    return sycl::rsqrt(a);
  }
};

template <typename T>
struct RsqrtFunctor<c10::complex<T>> {
  c10::complex<T> operator()(c10::complex<T> a) const {
    return c10::complex<T>(1.0, 0) /
        static_cast<c10::complex<T>>(
               std::sqrt(static_cast<std::complex<T>>(a)));
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
struct ExpFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::exp(static_cast<acc_t>(a));
  }
};

template <typename scalar_t>
struct BitwiseNotFunctor {
  scalar_t operator()(scalar_t a) const {
    return ~a;
  }
};

template <>
struct BitwiseNotFunctor<bool> {
  bool operator()(bool a) const {
    return !a;
  }
};

void sqrt_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "sqrt_xpu", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel(iter, SqrtFunctor<opmath_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "sqrt_xpu",
        [&]() { gpu_kernel(iter, SqrtFunctor<scalar_t>()); });
  }
}

void rsqrt_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "rsqrt_xpu", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel(iter, RsqrtFunctor<opmath_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::BFloat16,
        ScalarType::Half,
        iter.common_dtype(),
        "rsqrt_xpu",
        [&]() { gpu_kernel(iter, RsqrtFunctor<scalar_t>()); });
  }
}

void bitwise_not_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, BitwiseNotFunctor<bool>());
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_xpu", [&]() {
      gpu_kernel(iter, BitwiseNotFunctor<scalar_t>());
    });
  }
}

void exp_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "exp_xpu", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      auto caller = ExpFunctor<scalar_t, opmath_t>();
      gpu_kernel(iter, caller);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        common_dtype,
        "exp_xpu",
        [&]() {
          auto caller = ExpFunctor<scalar_t>();
          gpu_kernel(iter, caller);
        });
  }
}

template <typename scalar_t>
static inline scalar_t _nan_to_num_replace(
    scalar_t a,
    scalar_t nan_replacement,
    scalar_t pos_inf_replacement,
    scalar_t neg_inf_replacement) {
  return at::_isnan(a) ? nan_replacement
                       : (a == std::numeric_limits<scalar_t>::infinity()
                              ? pos_inf_replacement
                              : (a == -std::numeric_limits<scalar_t>::infinity()
                                     ? neg_inf_replacement
                                     : a));
}

template <typename scalar_t, typename value_t>
struct NanToNumComplexFunctor {
  scalar_t operator()(scalar_t a) const {
    value_t res_real = _nan_to_num_replace(
        a.real(), nan_replacement_, pos_inf_replacement_, neg_inf_replacement_);
    value_t res_imag = _nan_to_num_replace(
        a.imag(), nan_replacement_, pos_inf_replacement_, neg_inf_replacement_);
    return scalar_t(res_real, res_imag);
  }
  NanToNumComplexFunctor(
      value_t nan_replacement,
      value_t pos_inf_replacement,
      value_t neg_inf_replacement)
      : nan_replacement_(nan_replacement),
        pos_inf_replacement_(pos_inf_replacement),
        neg_inf_replacement_(neg_inf_replacement) {}

 private:
  value_t nan_replacement_;
  value_t pos_inf_replacement_;
  value_t neg_inf_replacement_;
};

template <typename scalar_t>
struct NanToNumFunctor {
  scalar_t operator()(scalar_t a) const {
    return _nan_to_num_replace(
        a, nan_replacement_, pos_inf_replacement_, neg_inf_replacement_);
  }
  NanToNumFunctor(
      scalar_t nan_replacement,
      scalar_t pos_inf_replacement,
      scalar_t neg_inf_replacement)
      : nan_replacement_(nan_replacement),
        pos_inf_replacement_(pos_inf_replacement),
        neg_inf_replacement_(neg_inf_replacement) {}

 private:
  scalar_t nan_replacement_;
  scalar_t pos_inf_replacement_;
  scalar_t neg_inf_replacement_;
};

void nan_to_num_kernel(
    TensorIteratorBase& iter,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf) {
  if (isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "nan_to_num_xpu", [&]() {
      using value_t = scalar_t::value_type;
      value_t nan_replacement = static_cast<value_t>(nan.value_or(0.));
      value_t pos_inf_replacement = pos_inf.has_value()
          ? static_cast<value_t>(pos_inf.value())
          : std::numeric_limits<value_t>::max();
      value_t neg_inf_replacement = neg_inf.has_value()
          ? static_cast<value_t>(neg_inf.value())
          : std::numeric_limits<value_t>::lowest();
      gpu_kernel(
          iter,
          NanToNumComplexFunctor<scalar_t, value_t>(
              nan_replacement, pos_inf_replacement, neg_inf_replacement));
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, iter.dtype(), "nan_to_num_xpu", [&]() {
          scalar_t nan_replacement = static_cast<scalar_t>(nan.value_or(0.));
          scalar_t pos_inf_replacement = pos_inf.has_value()
              ? static_cast<scalar_t>(pos_inf.value())
              : std::numeric_limits<scalar_t>::max();
          scalar_t neg_inf_replacement = neg_inf.has_value()
              ? static_cast<scalar_t>(neg_inf.value())
              : std::numeric_limits<scalar_t>::lowest();
          gpu_kernel(
              iter,
              NanToNumFunctor<scalar_t>(
                  nan_replacement, pos_inf_replacement, neg_inf_replacement));
        });
  }
}

template <typename scalar_t>
struct Expm1Functor {
  scalar_t operator()(scalar_t a) const {
    return std::expm1(a);
  }
};

template <typename T>
struct Expm1Functor<c10::complex<T>> {
  c10::complex<T> operator()(c10::complex<T> x) const {
    auto a = std::sin(T(.5) * x.imag());
    auto re = std::expm1(x.real()) * std::cos(x.imag()) - T(2) * a * a;
    auto im = std::exp(x.real()) * std::sin(x.imag());
    return c10::complex<T>(re, im);
  }
};

void expm1_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "expm1_xpu",
      [&]() { gpu_kernel(iter, Expm1Functor<scalar_t>()); });
}

template <typename scalar_t>
struct FrexpFunctor {
  std::tuple<scalar_t, int32_t> operator()(scalar_t a) const {
    int32_t exponent;
    scalar_t mantissa = std::frexp(a, &exponent);
    return {mantissa, exponent};
  }
};

void frexp_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      // The iter.dtype() here is the dtype of mantissa output.
      // It's a floating point type and must be the same as the input's dtype.
      iter.dtype(),
      "frexp_xpu",
      [&]() { gpu_kernel_multiple_outputs(iter, FrexpFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
