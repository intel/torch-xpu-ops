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

template <typename scalar_t>
struct FracFunctor {
  scalar_t operator()(scalar_t a) const {
    return a - std::trunc(a);
  }
};

void frac_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "frac_xpu", [&]() {
        gpu_kernel(iter, FracFunctor<scalar_t>());
      });
}

template <typename scalar_t>
struct CeilFunctor {
  scalar_t operator()(const scalar_t a) const {
    return std::ceil(a);
  }
};

template <typename T>
struct CeilFunctor<c10::complex<T>> {
  c10::complex<T> operator()(const c10::complex<T> a) const {
    return c10::complex<T>(std::ceil(a.real()), std::ceil(a.imag()));
  }
};

void ceil_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "ceil_xpu", [&]() {
        gpu_kernel(iter, CeilFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu
