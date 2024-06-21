#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Lerp.h>

namespace at {

template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE bool is_lerp_weight_small(scalar_t weight) {
  return std::abs(weight) < scalar_t(0.5);
}
template <typename scalar_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE bool is_lerp_weight_small(c10::complex<scalar_t> weight) {
  // Avoid the sqrt in abs(weight)
  return (weight.real() * weight.real() + weight.imag() * weight.imag()) < scalar_t(0.25);
}

template <typename scalar_t, typename weight_t>
C10_HOST_DEVICE C10_ALWAYS_INLINE scalar_t lerp(scalar_t self_, scalar_t end_, weight_t weight_) {
  using opmath_t = at::opmath_type<scalar_t>;
  using opmath_weight_t = at::opmath_type<weight_t>;

  opmath_t self = self_;
  opmath_t end = end_;
  opmath_weight_t weight = weight_;

  return is_lerp_weight_small(weight)
      ? self + weight * (end - self)
      : end - (end - self) * (opmath_t(1) - weight);
}

} // namespace at
