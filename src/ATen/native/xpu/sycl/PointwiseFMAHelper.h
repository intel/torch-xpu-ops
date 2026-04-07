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

#include <cmath>
#include <functional>
#include <type_traits>

namespace at::native::xpu {

// Computes input + alpha * op(tensor1, tensor2).
// Uses std::fma for real floating-point types to ensure consistent
// FMA behavior matching CUDA's pointwise_op_impl.
template <typename opmath_t, typename Op>
inline opmath_t pointwise_op_impl(
    opmath_t input,
    opmath_t tensor1,
    opmath_t tensor2,
    opmath_t alpha,
    Op op) {
  if (alpha == opmath_t(1)) {
    if constexpr (std::is_same_v<Op, std::multiplies<opmath_t>> &&
                  std::is_floating_point_v<opmath_t>) {
      return std::fma(tensor1, tensor2, input);
    } else {
      return input + op(tensor1, tensor2);
    }
  }
  if constexpr (std::is_floating_point_v<opmath_t>) {
    return std::fma(alpha, op(tensor1, tensor2), input);
  } else {
    return input + alpha * op(tensor1, tensor2);
  }
}

} // namespace at::native::xpu
