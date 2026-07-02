/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>

#include <c10/core/ScalarType.h>

#include <optional>

namespace at::native {

inline bool found_inf_nonzero(const std::optional<at::Tensor>& found_inf) {
  return found_inf.has_value() && found_inf->item<double>() != 0.0;
}

inline double grad_scale_value(const std::optional<at::Tensor>& grad_scale) {
  return grad_scale.has_value() && grad_scale->is_cpu()
      ? grad_scale->item<double>()
      : 1.0;
}

inline bool can_cast_without_narrowing(ScalarType from, ScalarType to) {
  return from == to ||
      (isFloatingType(from) && isFloatingType(to) &&
       promoteTypes(from, to) == to);
}

inline ScalarType fallback_math_dtype(
    ScalarType param_dtype,
    ScalarType grad_dtype,
    ScalarType exp_avg_dtype,
    ScalarType exp_avg_sq_dtype,
    std::optional<ScalarType> max_exp_avg_sq_dtype) {
  auto math_dtype = promoteTypes(param_dtype, grad_dtype);
  math_dtype = promoteTypes(math_dtype, exp_avg_dtype);
  math_dtype = promoteTypes(math_dtype, exp_avg_sq_dtype);
  if (max_exp_avg_sq_dtype.has_value()) {
    math_dtype = promoteTypes(math_dtype, *max_exp_avg_sq_dtype);
  }
  return isFloatingType(math_dtype) ? toOpMathType(math_dtype) : math_dtype;
}

} // namespace at::native
