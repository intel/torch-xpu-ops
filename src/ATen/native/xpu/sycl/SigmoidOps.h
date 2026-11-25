/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/OpMathType.h>
#include <c10/util/complex.h>

#ifndef SYCL_EXT_ONEAPI_COMPLEX
#define SYCL_EXT_ONEAPI_COMPLEX
#endif
#include <sycl/ext/oneapi/experimental/complex/complex.hpp>

namespace at::native::xpu {

template <typename scalar_t>
struct SigmoidFunctor {
  scalar_t operator()(scalar_t a) const {
    using opmath_t = at::opmath_type<scalar_t>;
    [[maybe_unused]] const auto one = opmath_t{1.0};
    const auto a_ = static_cast<opmath_t>(a);
    if constexpr (c10::is_complex<opmath_t>::value) {
      using value_t = typename opmath_t::value_type;
      static_assert(
          std::is_same_v<value_t, float> || std::is_same_v<value_t, double>,
          "Complex sigmoid requires float or double opmath type");
      // std::complex::exp on Windows/MSVC returns 0 instead of nan for
      // exp(finite + inf*i) if the real part exceeds some threshold;
      // the SYCL complex extension avoids this.
      namespace syclex = sycl::ext::oneapi::experimental;
      const syclex::complex<value_t> z(a_.real(), a_.imag());
      const auto e = syclex::exp(-z);
      return one / (one + opmath_t(e.real(), e.imag()));
    } else {
      return one / (one + sycl::exp(-a_));
    }
  }
};

} // namespace at::native::xpu
