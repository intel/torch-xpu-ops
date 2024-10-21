
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/complex.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/UnarySpecialOpsKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SigmoidFunctor {
  scalar_t operator()(scalar_t a) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto one = opmath_t{1.0};
    return one / (one + std::exp(-static_cast<opmath_t>(a)));
  }
};

void sigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      kComplexHalf,
      iter.common_dtype(),
      "sigmoid_xpu",
      [&]() { gpu_kernel(iter, SigmoidFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct ErfFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::erf(float(a));
  }
};

template <>
struct ErfFunctor<double> {
  double operator()(double a) const {
    return std::erf(a);
  }
};

template <typename scalar_t>
struct ErfcFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::erfc(float(a));
  }
};

template <>
struct ErfcFunctor<double> {
  double operator()(double a) const {
    return std::erfc(a);
  }
};

void erf_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "erf_xpu",
      [&]() { gpu_kernel(iter, ErfFunctor<scalar_t>()); });
}

void erfc_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "erfc_xpu",
      [&]() { gpu_kernel(iter, ErfcFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct ErfinvFunctor {
  scalar_t operator()(scalar_t in) const {
    return calc_erfinv(in);
  }
};

template <>
struct ErfinvFunctor<c10::Half> {
  c10::Half operator()(c10::Half in) const {
    return calc_erfinv(float(in));
  }
};

void erfinv_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "erfinv_xpu",
      [&]() { gpu_kernel(iter, ErfinvFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct Exp2Functor {
  scalar_t operator()(scalar_t a) const {
    return std::exp2(a);
  }
};

template <typename T>
struct Exp2Functor<c10::complex<T>> {
  c10::complex<T> operator()(c10::complex<T> x) const {
    // There is no std::exp2 overload for complex, so instead
    // use the identity 2^x = e^(ln(2) * x)
    const auto ln_2 = static_cast<T>(0.693147180559945309417232121458176);
    return std::exp(ln_2 * x);
  }
};

void exp2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "exp2_xpu",
      [&]() { gpu_kernel(iter, Exp2Functor<scalar_t>()); });
}

template <typename scalar_t>
struct Logit0Functor {
  using T_ACC = acc_type_device<scalar_t, c10::DeviceType::XPU>;
  scalar_t operator()(scalar_t x) const {
    const T_ACC x_acc = static_cast<T_ACC>(x);
    // suppress compiler optimization on data type promotion.
    volatile T_ACC res = std::log(x_acc / (T_ACC(1) - x_acc));
    return res;
  }
};

template <typename scalar_t>
struct Logit1Functor {
  using T_ACC = acc_type_device<scalar_t, c10::DeviceType::XPU>;
  scalar_t operator()(scalar_t x) const {
    const T_ACC x_acc = static_cast<T_ACC>(x);
    T_ACC z = x_acc < lo_ ? lo_ : (x_acc > hi_ ? hi_ : x_acc);
    // suppress compiler optimization on data type promotion.
    volatile T_ACC res = std::log(z / (T_ACC(1) - z));
    return res;
  }
  Logit1Functor(const T_ACC lo, const T_ACC hi) : lo_(lo), hi_(hi) {}

 private:
  T_ACC lo_;
  T_ACC hi_;
};

void logit_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "logit_xpu",
      [&]() {
        using T_ACC = acc_type_device<scalar_t, c10::DeviceType::XPU>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          gpu_kernel(iter, Logit0Functor<scalar_t>());
        } else {
          const T_ACC lo = eps;
          const T_ACC hi = T_ACC(1) - eps;
          gpu_kernel(iter, Logit1Functor<scalar_t>(lo, hi));
        }
      });
}

} // namespace at::native::xpu
