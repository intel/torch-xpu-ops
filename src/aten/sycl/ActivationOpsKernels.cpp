#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct ReluFunctor {
  scalar_t operator()(scalar_t x) const {
    return x <= scalar_t{0} ? scalar_t{0} : x;
  }
};

template <typename scalar_t>
struct ThresholdFunctor {
  scalar_t operator()(scalar_t x, scalar_t other) const {
    return x <= threshold_ ? value_ : other;
  }

  ThresholdFunctor(scalar_t threshold, scalar_t value)
      : threshold_(threshold), value_(value) {}

 private:
  scalar_t threshold_;
  scalar_t value_;
};

// constexpr float M_SQRT2 = 1.41421356237309504880;
// constexpr float M_2_SQRTPI = 1.12837916709551257390;
// constexpr float M_SQRT1_2 = 0.70710678118654752440;

template <typename scalar_t>
struct GeluTanhFunctor {
  scalar_t operator()(scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
    constexpr opmath_t kKappa = 0.044715;
    auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) *
        static_cast<opmath_t>(x);
    auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
    return opmath_t(0.5) * static_cast<opmath_t>(x) *
        (opmath_t(1) + ::tanh(inner));
  }
};

template <typename scalar_t>
struct GeluTanhBackwardFunctor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
    constexpr opmath_t kKappa = 0.044715;
    auto x_sq = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
    auto x_cube = x_sq * static_cast<opmath_t>(x);
    auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
    auto tanh_inner = ::tanh(inner);

    auto left = opmath_t(0.5) * static_cast<opmath_t>(x);
    auto right = opmath_t(1) + tanh_inner;

    auto left_derivative = 0.5 * right;

    auto tanh_derivative = opmath_t(1) - tanh_inner * tanh_inner;
    auto inner_derivative = kBeta * (opmath_t(1) + opmath_t(3) * kKappa * x_sq);
    auto right_derivative = left * tanh_derivative * inner_derivative;

    return static_cast<opmath_t>(dy) * (left_derivative + right_derivative);
  }
};

template <typename scalar_t>
struct GeluErfFunctor {
  scalar_t operator()(scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    constexpr opmath_t kAlpha = M_SQRT1_2;
    return static_cast<opmath_t>(x) * opmath_t(0.5) *
        (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
  }
};

template <typename scalar_t>
struct GeluErfBackwardFunctor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * opmath_t(0.5);
    constexpr opmath_t kAlpha = M_SQRT1_2;
    const opmath_t cdf = opmath_t(0.5) *
        (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
    const opmath_t pdf = ::exp(
                             opmath_t(-0.5) * static_cast<opmath_t>(x) *
                             static_cast<opmath_t>(x)) *
        kBeta;
    return static_cast<opmath_t>(dy) * (cdf + static_cast<opmath_t>(x) * pdf);
  }
};

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

void relu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "relu_xpu", [&]() {
    gpu_kernel(iter, ReluFunctor<scalar_t>());
  });
}

void threshold_kernel(
    TensorIteratorBase& iter,
    const Scalar& threshold,
    const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "threshold_xpu", [&]() {
        scalar_t threshold_ = threshold.to<scalar_t>();
        scalar_t value_ = value.to<scalar_t>();
        gpu_kernel(iter, ThresholdFunctor<scalar_t>(threshold_, value_));
      });
}

void gelu_kernel(TensorIteratorBase& iter, c10::string_view approximate) {
  auto approximate_ = at::native::get_gelutype_enum(approximate);
  if (approximate_ == at::native::GeluType::Tanh) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "gelu_tanh_xpu",
        [&]() { gpu_kernel(iter, GeluTanhFunctor<scalar_t>()); });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "gelu_erf_xpu",
        [&]() { gpu_kernel(iter, GeluErfFunctor<scalar_t>()); });
  }
}

void gelu_backward_kernel(
    TensorIteratorBase& iter,
    c10::string_view approximate) {
  auto approximate_ = at::native::get_gelutype_enum(approximate);
  if (approximate_ == at::native::GeluType::Tanh) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "gelu_tanh_backward_xpu",
        [&]() {
          gpu_kernel_with_scalars(iter, GeluTanhBackwardFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::BFloat16,
        at::ScalarType::Half,
        iter.dtype(),
        "gelu_erf_backward_xpu",
        [&]() {
          gpu_kernel_with_scalars(iter, GeluErfBackwardFunctor<scalar_t>());
        });
  }
}

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

} // namespace xpu
} // namespace native
} // namespace at
