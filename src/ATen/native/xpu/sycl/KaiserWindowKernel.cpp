#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/KaiserWindowKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MathExtensions.h>

namespace at::native::xpu {

template <typename scalar_t>
struct KaiserWindowFunctor {
  scalar_t operator()(scalar_t a) const {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_t x = static_cast<opmath_t>(a) * inv_alpha_ - opmath_t(1);
    opmath_t y = std::max(opmath_t(0), opmath_t(1) - x * x);
    return calc_i0(static_cast<opmath_t>(beta_) * std::sqrt(y)) * inv_i0_beta_;
  }
  KaiserWindowFunctor(double beta, double inv_alpha, double inv_i0_beta)
      : beta_(beta), inv_alpha_(inv_alpha), inv_i0_beta_(inv_i0_beta) {}

 private:
  double beta_;
  double inv_alpha_;
  double inv_i0_beta_;
};

void kaiser_window_kernel(TensorIteratorBase& iter, int64_t window_length, double beta) {
  const auto inv_alpha = 2.0 / (window_length - 1.0);
  const auto inv_i0_beta = 1.0 / calc_i0(beta);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "kaiser_window_xpu",
      [&]() {
        gpu_kernel(
            iter, KaiserWindowFunctor<scalar_t>(beta, inv_alpha, inv_i0_beta));
      });
}

} // namespace at::native::xpu
