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
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct WhereFunctor {
  scalar_t operator()(bool cond_val, scalar_t self_val, scalar_t other_val)
      const {
    return cond_val ? self_val : other_val;
  }
};

template <typename scalar_t>
struct ClampFunctor {
  scalar_t operator()(scalar_t v, scalar_t lower, scalar_t upper) const {
    if (at::_isnan(v)) {
      return v;
    }
    if (at::_isnan(lower)) {
      return lower;
    }
    if (at::_isnan(upper)) {
      return upper;
    } else {
      return std::min(std::max(v, lower), upper);
    }
  }
};

template <typename scalar_t>
struct ClampScalarFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t v) const {
    if (_isnan(static_cast<opmath_t>(v))) {
      return v;
    } else if (minmax_ == at::native::detail::ClampLimits::Min) {
      return std::max(static_cast<opmath_t>(v), lim0_val_);
    } else if (minmax_ == at::native::detail::ClampLimits::Max) {
      return std::min(static_cast<opmath_t>(v), lim0_val_);
    } else {
      return std::min(std::max(static_cast<opmath_t>(v), lim0_val_), lim1_val_);
    }
  }
  ClampScalarFunctor(
      opmath_t lim0_val,
      opmath_t lim1_val,
      at::native::detail::ClampLimits minmax)
      : lim0_val_(lim0_val), lim1_val_(lim1_val), minmax_(minmax) {}

 private:
  opmath_t lim0_val_;
  opmath_t lim1_val_;
  at::native::detail::ClampLimits minmax_;
};

void inline launch_clamp_scalar(
    TensorIteratorBase& iter,
    Scalar lim0,
    Scalar lim1,
    at::native::detail::ClampLimits minmax) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "clamp_scalar_xpu", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        auto lim0_val = lim0.to<opmath_t>();
        auto lim1_val = lim1.to<opmath_t>();
        gpu_kernel(
            iter, ClampScalarFunctor<scalar_t>(lim0_val, lim1_val, minmax));
      });
}

} // namespace xpu
} // namespace native
} // namespace at
