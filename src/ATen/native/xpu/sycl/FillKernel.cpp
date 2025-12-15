/*
 * Copyright 2020-2025 Intel Corporation
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
        gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
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
