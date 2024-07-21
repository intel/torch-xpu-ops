#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/util/MathConstants.h>
namespace at::native::xpu {

template <typename scalar_t>
struct LogAddExpFunctor {
  scalar_t operator()(scalar_t a_, scalar_t b_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto a = static_cast<opmath_t>(a_);
    const auto b = static_cast<opmath_t>(b_);
    if (std::isinf(a) && a == b) {
      return a;
    } else {
      const auto m = std::max(a, b);
      return m + std::log1p(std::exp(-std::abs(a - b)));
    }
  }
};

void logaddexp_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.dtype(),
      "logaddexp_xpu",
      [&]() { gpu_kernel(iter, LogAddExpFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct LogAddExp2Functor {
  scalar_t operator()(scalar_t a_, scalar_t b_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto inv_log_2 = static_cast<opmath_t>(1.0 / c10::ln_2<double>);
    const auto a = static_cast<opmath_t>(a_);
    const auto b = static_cast<opmath_t>(b_);
    if (std::isinf(a) && a == b) {
      return a;
    } else {
      const auto m = std::max(a, b);
      return m + std::log1p(std::exp2(-std::abs(a - b))) * inv_log_2;
    }
  }
};

void logaddexp2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.dtype(),
      "logaddexp2_xpu",
      [&]() { gpu_kernel(iter, LogAddExp2Functor<scalar_t>()); });
}

} // namespace at::native::xpu
