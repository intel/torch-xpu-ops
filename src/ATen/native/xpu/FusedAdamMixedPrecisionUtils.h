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

#include <ATen/core/Tensor.h>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

namespace at::native {

inline void validate_fused_mixed_precision_dtypes(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    const char* op_name) {
  TORCH_CHECK(
      params[0].scalar_type() == at::kFloat,
      op_name,
      " with mixed dtypes requires float32 params, got ",
      params[0].scalar_type());
  TORCH_CHECK(
      grads[0].scalar_type() == at::kFloat,
      op_name,
      " with mixed dtypes requires float32 grads, got ",
      grads[0].scalar_type());
  TORCH_CHECK(
      exp_avgs[0].scalar_type() == at::kBFloat16,
      op_name,
      " with mixed dtypes requires bfloat16 optimizer states, got ",
      exp_avgs[0].scalar_type());
  TORCH_CHECK(
      exp_avg_sqs[0].scalar_type() == at::kBFloat16,
      op_name,
      " with mixed dtypes requires bfloat16 optimizer states, got ",
      exp_avg_sqs[0].scalar_type());
}

inline void validate_fused_mixed_precision_dtypes(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    const char* op_name) {
  validate_fused_mixed_precision_dtypes(
      params, grads, exp_avgs, exp_avg_sqs, op_name);
  TORCH_CHECK(
      max_exp_avg_sqs[0].scalar_type() == at::kBFloat16,
      op_name,
      " with mixed dtypes requires bfloat16 max_exp_avg_sqs, got ",
      max_exp_avg_sqs[0].scalar_type());
}

} // namespace at::native
