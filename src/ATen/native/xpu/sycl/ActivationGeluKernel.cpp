#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/XPUMathCompat.h>

namespace at {
namespace native {
namespace xpu {

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
        (opmath_t(1) + c10::xpu::compat::tanh(inner));
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
    auto tanh_inner = c10::xpu::compat::tanh(inner);

    auto left = opmath_t(0.5) * static_cast<opmath_t>(x);
    auto right = opmath_t(1) + tanh_inner;

    auto left_derivative = opmath_t(0.5) * right;

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
        (opmath_t(1) + std::erf(static_cast<opmath_t>(x) * kAlpha));
  }
};

template <typename scalar_t>
struct GeluErfBackwardFunctor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * opmath_t(0.5);
    constexpr opmath_t kAlpha = M_SQRT1_2;
    const opmath_t cdf = opmath_t(0.5) *
        (opmath_t(1) + std::erf(static_cast<opmath_t>(x) * kAlpha));
    const opmath_t pdf = c10::xpu::compat::exp(
                             opmath_t(-0.5) * static_cast<opmath_t>(x) *
                             static_cast<opmath_t>(x)) *
        kBeta;
    return static_cast<opmath_t>(dy) * (cdf + static_cast<opmath_t>(x) * pdf);
  }
};

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

} // namespace xpu
} // namespace native
} // namespace at
