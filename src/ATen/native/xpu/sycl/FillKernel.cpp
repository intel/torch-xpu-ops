/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch_v2.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/FillKernel.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct FillFunctor {
  scalar_t operator()() const {
    return val_;
  }
  FillFunctor(scalar_t val) : val_(val) {}

 private:
  scalar_t val_;
};

void fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_V2(
      iter.dtype(),
      "fill_xpu",
      AT_WRAP([&]() {
        // For reduced-precision float types, use static_cast via double to allow
        // out-of-range values to saturate to ±inf (matching CPU behavior in
        // aten/src/ATen/ScalarOps.cpp fill_inplace), rather than throwing a
        // RuntimeError on overflow.
        scalar_t val;
        if constexpr (
            std::is_same_v<scalar_t, at::Half> ||
            std::is_same_v<scalar_t, at::BFloat16> ||
            std::is_same_v<scalar_t, at::Float8_e4m3fn> ||
            std::is_same_v<scalar_t, at::Float8_e5m2> ||
            std::is_same_v<scalar_t, at::Float8_e4m3fnuz> ||
            std::is_same_v<scalar_t, at::Float8_e5m2fnuz>) {
          val = static_cast<scalar_t>(value.to<double>());
        } else {
          val = value.to<scalar_t>();
        }
        gpu_kernel(iter, FillFunctor<scalar_t>(val));
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      kComplexHalf,
      kBool,
      kHalf,
      kBFloat16,
      AT_EXPAND(AT_FLOAT8_TYPES),
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

} // namespace xpu
} // namespace native
} // namespace at
