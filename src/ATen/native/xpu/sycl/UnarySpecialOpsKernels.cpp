#include <ATen/ATen.h>
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
struct Logit0Functor {
  scalar_t operator()(scalar_t x) const {
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC x_acc = static_cast<T_ACC>(x);
    return std::log(x_acc / (T_ACC(1) - x_acc));
  }
};

template <typename scalar_t>
struct Logit1Functor {
  scalar_t operator()(scalar_t x) const {
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC lo = eps_;
    const T_ACC hi = T_ACC(1) - eps_;
    const T_ACC x_acc = static_cast<T_ACC>(x);
    T_ACC z = x_acc < lo ? lo : (x_acc > hi ? hi : x_acc);
    return std::log(z / (T_ACC(1) - z));
  }
  Logit1Functor(const scalar_t eps) : eps_(eps) {}

 private:
  using T_ACC = acc_type<scalar_t, true>;
  T_ACC eps_;
};

void logit_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "logit_xpu",
      [&]() {
        using T_ACC = acc_type<scalar_t, true>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          gpu_kernel(iter, Logit0Functor<scalar_t>());
        } else {
          gpu_kernel(iter, Logit1Functor<scalar_t>(eps));
        }
      });
}

} // namespace at::native::xpu
