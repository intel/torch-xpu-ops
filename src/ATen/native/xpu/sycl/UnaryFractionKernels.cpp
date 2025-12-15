/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/UnaryFractionKernels.h>

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

template <typename scalar_t>
inline scalar_t nearbyint_wrapper(scalar_t a) {
  return static_cast<scalar_t>(std::nearbyintf(static_cast<float>(a)));
}

inline double nearbyint_wrapper(double a) {
  return std::nearbyint(a);
}

#pragma push
inline c10::complex<float> nearbyint_wrapper(c10::complex<float> a) {
  return c10::complex<float>(
      std::nearbyintf(static_cast<float>(a.real())),
      std::nearbyintf(static_cast<float>(a.imag())));
}

inline c10::complex<double> nearbyint_wrapper(c10::complex<double> a) {
  return c10::complex<double>(
      std::nearbyint(static_cast<double>(a.real())),
      std::nearbyint(static_cast<double>(a.imag())));
}
#pragma pop

template <typename scalar_t>
struct RoundFunctor {
  scalar_t operator()(scalar_t a) const {
    return nearbyint_wrapper(a);
  }
};

template <typename scalar_t>
struct RoundDecimalsFunctor {
  scalar_t operator()(scalar_t a) const {
    return neg_flag_
        ? std::nearbyint(a / ten_pow_decimals_) * ten_pow_decimals_
        : std::nearbyint(a * ten_pow_decimals_) / ten_pow_decimals_;
  }
  RoundDecimalsFunctor(scalar_t ten_pow_decimals, bool neg_flag)
      : ten_pow_decimals_(ten_pow_decimals), neg_flag_(neg_flag) {}

 private:
  scalar_t ten_pow_decimals_;
  bool neg_flag_;
};

void round_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "round_xpu", [&]() {
        gpu_kernel(iter, RoundFunctor<scalar_t>());
      });
}

void round_decimals_kernel(TensorIteratorBase& iter, int64_t decimals) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "round_xpu", [&]() {
        bool neg_flag = false;
        scalar_t ten_pow_decimals;
        if (decimals < 0) {
          decimals = -decimals;
          neg_flag = true;
        }
        ten_pow_decimals = static_cast<scalar_t>(std::pow(10, decimals));
        gpu_kernel(
            iter, RoundDecimalsFunctor<scalar_t>(ten_pow_decimals, neg_flag));
      });
}

template <typename scalar_t>
struct FloorFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::floor(a);
  }
};

template <typename T>
struct FloorFunctor<c10::complex<T>> {
  c10::complex<T> operator()(c10::complex<T> v) const {
    return c10::complex<T>(std::floor(v.real()), std::floor(v.imag()));
  }
};

void floor_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "floor_xpu", [&]() {
        gpu_kernel(iter, FloorFunctor<scalar_t>());
      });
}

// We manually overload trunc because std::trunc does not work with std::complex
// types and ROCm.
template <typename scalar_t>
inline scalar_t trunc_wrapper(scalar_t a) {
  return static_cast<scalar_t>(std::truncf(static_cast<float>(a)));
}

inline double trunc_wrapper(double a) {
  return std::trunc(a);
}

inline c10::complex<float> trunc_wrapper(c10::complex<float> a) {
  return c10::complex<float>(
      std::truncf(static_cast<float>(a.real())),
      std::truncf(static_cast<float>(a.imag())));
}

inline c10::complex<double> trunc_wrapper(c10::complex<double> a) {
  return c10::complex<double>(
      std::trunc(static_cast<double>(a.real())),
      std::trunc(static_cast<double>(a.imag())));
}

template <typename scalar_t>
struct TruncFunctor {
  scalar_t operator()(scalar_t a) const {
    return trunc_wrapper(a);
  }
};

void trunc_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "trunc_xpu", [&]() {
        gpu_kernel(iter, TruncFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu
