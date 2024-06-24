#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
inline scalar_t reciprocal_wrapper(scalar_t a) {
  return static_cast<scalar_t>(1) / a;
}

template <typename T>
inline c10::complex<T> reciprocal_wrapper(c10::complex<T> v) {
  // Handle extreme cases for numpy compatibility
  auto both_inf = [](T real, T imag) {
    return (std::isinf(real) && std::isinf(imag));
  };

  auto either_inf = [](T real, T imag) {
    return std::isinf(real) || std::isinf(imag);
  };

  auto either_nan = [](T real, T imag) {
    return std::isnan(real) || std::isnan(imag);
  };

  if (either_nan(v.real(), v.imag()) || both_inf(v.real(), v.imag())) {
    // If either is Nan or both are infinite, return {nan, nan}
    return {
        std::numeric_limits<T>::quiet_NaN(),
        std::numeric_limits<T>::quiet_NaN()};
  } else if (either_inf(v.real(), v.imag())) {
    // If either is Inf, return {0, 0}
    return {0, 0};
  }
  const c10::complex<T> one = c10::complex<T>(1.0, 0);
  return one / v;
}

template <typename scalar_t>
struct ReciprocalFunctor {
  scalar_t operator()(scalar_t a) const {
    return reciprocal_wrapper<scalar_t>(a);
  }
};

void reciprocal_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "reciprocal_xpu",
      [&]() { gpu_kernel(iter, ReciprocalFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
