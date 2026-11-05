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

#include <ATen/core/Tensor.h>
#include <ATen/xpu/PhiloxXpuState.h>

namespace at::native::xpu {

TORCH_XPU_API void _transform_bias_rescale_qkv_kernel(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head,
    Tensor& q_k_v,
    int64_t B,
    int64_t T,
    int64_t D,
    int64_t dim_per_head);

TORCH_XPU_API void unpack_philox_kernel(
    at::PhiloxXpuState arg,
    int64_t* seed_ptr,
    int64_t* offset_ptr);

} // namespace at::native::xpu
