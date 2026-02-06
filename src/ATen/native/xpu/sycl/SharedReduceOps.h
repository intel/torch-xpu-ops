/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <ATen/NumericUtils.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/cpu/zmath.h>
#include <c10/macros/Macros.h>
#include <comm/XPUPair.h>
#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>

#define MAX(X, Y) max_impl(X, Y)
#define MIN(X, Y) min_impl(X, Y)

#define device_sqrt std::sqrt
#define compat_pow std::pow

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, typename index_t>
struct WelfordData {
  scalar_t first_elem;
  scalar_t sum;
  scalar_t sum_of_squares;
  index_t n;
  scalar_t nf;

  WelfordData() : first_elem(NULL), sum(0), sum_of_squares(0), n(0), nf(0) {}

  WelfordData(scalar_t first_elem, scalar_t sum, scalar_t sum_of_squares, index_t n, scalar_t nf)
      : first_elem(first_elem), sum(sum), sum_of_squares(sum_of_squares), n(n), nf(nf) {}
};

template <
    typename scalar_t,
    typename acc_scalar_t,
    typename index_t,
    typename res_t>
struct WelfordOps {
  acc_scalar_t correction;
  bool take_sqrt;

public:
  using acc_t = WelfordData<acc_scalar_t, index_t>;
  inline acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    // We accumulate n in index_t to avoid cumulative rounding error, but still
    // need nf for use in combine where int32 may overflow.
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    acc_scalar_t first_elem = acc.first_elem;
    acc_scalar_t sum = acc.sum + data;
    acc_scalar_t dif = static_cast<acc_scalar_t>(data) - acc.first_elem;
    acc_scalar_t sum_of_squares = acc.sum_of_squares + dif*dif;

    return {
        first_elem,
        sum,
        sum_of_squares,
        new_n,
        new_nf,
    };
  }
  inline acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t new_count = a.nf + b.nf;
    return {
      b.first_elem,
      a.sum + b.sum,
      a.sum_of_squares + b.sum_of_squares,
      a.n+b.n,
      new_count};
  }
  inline res_t project(acc_t acc) const __ubsan_ignore_float_divide_by_zero__ {
    const auto mean = static_cast<scalar_t>(acc.sum/acc.nf);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = (acc.sum_of_squares/acc.nf) - (mean - acc.first_elem) * (mean - acc.first_elem);
    res_t results(take_sqrt ? device_sqrt(var) : var, mean);
    return results;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

  WelfordOps(acc_scalar_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}
};

template <
    typename scalar_t,
    typename acc_t = scalar_t,
    typename factor_t = acc_t,
    typename out_t = acc_t>
struct MeanOps {
  factor_t factor;

  inline acc_t reduce(acc_t a, scalar_t b, int64_t /*idx*/) const {
    return combine(a, static_cast<acc_t>(b));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline out_t project(acc_t a) const {
    return a * factor;
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

  MeanOps(factor_t factor) : factor(factor) {}
};

// This accumulator template is used to calculate the minimum absolute value of
// a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct AbsMinOps {
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return MIN(acc, static_cast<acc_t>(std::abs(data)));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return MIN(a, b);
  }

  inline out_t project(acc_t a) const {
    return a;
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

// This accumulator template is used to calculate the maximum absolute value of
// a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct AbsMaxOps {
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return MAX(acc, static_cast<acc_t>(std::abs(data)));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return MAX(a, b);
  }

  inline out_t project(acc_t a) const {
    return a;
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

// This accumulator template is used to calculate the norm of the absolute value
// of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct NormOps {
  acc_t norm_;

  inline acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc + compat_pow(static_cast<acc_t>(std::abs(data)), norm_);
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline out_t project(acc_t a) const {
    return compat_pow(a, static_cast<acc_t>(1.0) / norm_);
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }

  NormOps(acc_t norm_) : norm_(norm_) {}
};

// This accumulator template is used to calculate the order zero norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct NormZeroOps {
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc +
        (data == static_cast<scalar_t>(0) ? static_cast<acc_t>(0)
                                          : static_cast<acc_t>(1));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline out_t project(acc_t a) const {
    return a;
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

// This accumulator template is used to calculate the order one norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct NormOneOps {
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    return acc + static_cast<acc_t>(std::abs(data));
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline out_t project(acc_t a) const {
    return a;
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

template <typename acc_t>
struct AbsSwitch {};

template <typename scalar_t, typename acc_t>
inline acc_t abs_if_complex(scalar_t data, AbsSwitch<acc_t>) {
  return static_cast<acc_t>(data);
}

template <typename scalar_t, typename acc_t>
inline acc_t abs_if_complex(std::complex<scalar_t> data, AbsSwitch<acc_t>) {
  return static_cast<acc_t>(std::abs(data));
}

template <typename scalar_t, typename acc_t>
inline acc_t abs_if_complex(c10::complex<scalar_t> data, AbsSwitch<acc_t>) {
  return static_cast<acc_t>(std::abs(data));
}

// This accumulator template is used to calculate the order two norm of the
// absolute value of a set of numbers.
// `scalar_t` is the type of the input and `acc_t` is the type of the
// accumulated value. These types differ for complex number input support.
template <typename scalar_t, typename acc_t = scalar_t, typename out_t = acc_t>
struct NormTwoOps {
  inline acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    acc_t data_ = abs_if_complex(data, AbsSwitch<acc_t>());
    return acc + data_ * data_;
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline out_t project(acc_t a) const {
    return device_sqrt(a);
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

template <typename acc_t, typename data_t>
struct NanSumOps {
  inline acc_t reduce(acc_t a, data_t b, int64_t /*idx*/) const {
    return a + (at::_isnan(b) ? acc_t{0.} : acc_t{b});
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    return a + b;
  }

  inline data_t project(acc_t a) const {
    return data_t{a};
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

namespace detail {

template <typename scalar_t>
struct LessOrNan {
  bool operator()(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
    // If (a == b), then choose the one with lower idx, else min(a, b)
    if (at::_isnan(a)) {
      if (at::_isnan(b)) {
        return idx_a < idx_b;
      }
      return true;
    }
    return (a == b) ? idx_a < idx_b : (a < b);
  }
};

template <typename scalar_t>
struct GreaterOrNan {
  bool operator()(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
    // If (a == b), then choose the one with lower idx, else max(a, b)
    if (at::_isnan(a)) {
      if (at::_isnan(b)) {
        return idx_a < idx_b;
      }
      return true;
    }
    return (a == b) ? idx_a < idx_b : (a > b);
  }
};

template <typename comp_t>
struct MinMaxReductionOps {
  using scalar_t = typename binary_function_traits<comp_t>::arg1_t;
  using index_t = int64_t;
  using arg_t = at::xpu::pair<scalar_t, index_t>;

  static arg_t project(arg_t arg) {
    return arg;
  }

  static arg_t reduce(arg_t arg, scalar_t val, int64_t idx) {
    return comp_t{}(arg.first, val, arg.second, idx) ? arg : arg_t(val, idx);
  }

  static arg_t combine(arg_t a, arg_t b) {
    return comp_t{}(a.first, b.first, a.second, b.second) ? a : b;
  }

  static arg_t translate_idx(arg_t a, int64_t base_idx) {
    return {a.first, a.second + base_idx};
  }
};

template <typename comp_t>
struct ArgReductionOps : public MinMaxReductionOps<comp_t> {
  using typename MinMaxReductionOps<comp_t>::scalar_t;
  using typename MinMaxReductionOps<comp_t>::index_t;
  using typename MinMaxReductionOps<comp_t>::arg_t;

  static index_t project(arg_t arg) {
    return arg.second;
  }
};

} // namespace detail

template <typename scalar_t>
struct ArgMaxOps
    : public detail::ArgReductionOps<detail::GreaterOrNan<scalar_t>> {};

template <typename scalar_t>
struct ArgMinOps : public detail::ArgReductionOps<detail::LessOrNan<scalar_t>> {
};

template <typename scalar_t>
struct MinOps : public detail::MinMaxReductionOps<detail::LessOrNan<scalar_t>> {
};

template <typename scalar_t>
struct MaxOps
    : public detail::MinMaxReductionOps<detail::GreaterOrNan<scalar_t>> {};

template <typename scalar_t, typename acc_scalar_t, typename index_t>
struct MinMaxOps {
  using acc_t = at::xpu::pair<acc_scalar_t, acc_scalar_t>;
  inline acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    return combine(acc, {data, data});
  }

  inline acc_t combine(acc_t a, acc_t b) const {
    auto min_val =
        (at::_isnan(a.first) || a.first < b.first) ? a.first : b.first;
    auto max_val =
        (at::_isnan(a.second) || a.second > b.second) ? a.second : b.second;

    return {min_val, max_val};
  }

  inline acc_t project(acc_t acc) const {
    return acc;
  }

  static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

} // namespace xpu
} // namespace native
} // namespace at

#undef MAX
#undef MIN
