#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/KaiserWindowKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MathExtensions.h>

namespace at::native::xpu {

template <typename scalar_t>
struct KaiserWindowFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  
  scalar_t operator()(scalar_t a) const {
    opmath_t norm = static_cast<opmath_t>(a) * inv_alpha_ - 1;
    opmath_t sqrt_term = std::max<opmath_t>(0, 1 - norm * norm);
    return calc_i0(beta_ * std::sqrt(sqrt_term)) * inv_i0_beta_;
  }
  
  KaiserWindowFunctor(opmath_t beta, opmath_t inv_alpha, opmath_t inv_i0_beta)
      : beta_(beta), inv_alpha_(inv_alpha), inv_i0_beta_(inv_i0_beta) {}

 private:
  opmath_t beta_;
  opmath_t inv_alpha_;
  opmath_t inv_i0_beta_;
};

void kaiser_window_kernel(TensorIteratorBase& iter, int64_t window_length, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "kaiser_window_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t inv_alpha = static_cast<opmath_t>(2.0 / (window_length - 1));
        const opmath_t beta_opmath = static_cast<opmath_t>(beta);
        const opmath_t inv_i0_beta = 1.0 / calc_i0(beta_opmath);
        gpu_kernel(
            iter, KaiserWindowFunctor<scalar_t>(beta_opmath, inv_alpha, inv_i0_beta));
      });
}

} // namespace at::native::xpu
