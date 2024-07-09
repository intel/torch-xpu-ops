#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SigmoidBackwardComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto one = opmath_t{1.};
    const auto comp_b = static_cast<opmath_t>(b);
    const auto comp_a = static_cast<opmath_t>(a);
    return static_cast<scalar_t>(comp_a * std::conj((one - comp_b) * comp_b));
  }
};

template <typename scalar_t>
struct SigmoidBackwardFloatFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * (scalar_t(1.) - b) * b;
  }
};

void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, dtype, "sigmoid_backward_xpu", [&]() {
          gpu_kernel(iter, SigmoidBackwardComplexFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "sigmoid_backward_xpu",
        [&]() { gpu_kernel(iter, SigmoidBackwardFloatFunctor<scalar_t>()); });
  }
}

template <typename scalar_t>
struct LogitBackward0Functor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC dy_acc = static_cast<T_ACC>(dy);
    const T_ACC x_acc = static_cast<T_ACC>(x);
    return (x_acc < T_ACC(0) || x_acc > T_ACC(1))
        ? std::numeric_limits<T_ACC>::quiet_NaN()
        : dy_acc / (x_acc * (T_ACC(1) - x_acc));
  }
};

template <typename scalar_t>
struct LogitBackward1Functor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC lo = eps_;
    const T_ACC hi = T_ACC(1) - eps_;
    const T_ACC dy_acc = static_cast<T_ACC>(dy);
    const T_ACC x_acc = static_cast<T_ACC>(x);
    return (x_acc < lo || x_acc > hi) ? T_ACC(0)
                                      : dy_acc / (x_acc * (T_ACC(1) - x_acc));
  }
  LogitBackward1Functor(const scalar_t eps) : eps_(eps) {}

 private:
  using T_ACC = acc_type<scalar_t, true>;
  T_ACC eps_;
};

void logit_backward_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "logit_xpu",
      [&]() {
        using T_ACC = acc_type<scalar_t, true>;
        const T_ACC eps = eps_scalar.to<T_ACC>();
        if (eps < T_ACC(0)) {
          gpu_kernel(iter, LogitBackward0Functor<scalar_t>());
        } else {
          gpu_kernel(iter, LogitBackward1Functor<scalar_t>(eps));
        }
      });
}

template <typename scalar_t>
struct TanhBackwardComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    using comp_t = at::opmath_type<scalar_t>;
    const auto one = comp_t{1.};
    const auto comp_b = static_cast<comp_t>(b);
    const auto comp_a = static_cast<comp_t>(a);
    return static_cast<scalar_t>(comp_a * std::conj(one - comp_b * comp_b));
  }
};

template <typename scalar_t>
struct TanhBackwardFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * (scalar_t{1.} - b * b);
  }
};

void tanh_backward_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, dtype, "tanh_backward_complex_xpu", [&]() {
          gpu_kernel(iter, TanhBackwardComplexFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "tanh_backward_xpu",
        [&]() { gpu_kernel(iter, TanhBackwardFunctor<scalar_t>()); });
  }
}

} // namespace at::native::xpu
