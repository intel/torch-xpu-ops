/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor> weight_norm_kernel(
    const Tensor& v,
    const Tensor& g,
    int64_t dim);

TORCH_XPU_API std::tuple<Tensor, Tensor> weight_norm_backward_kernel(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norms,
    int64_t dim);

} // namespace at::native::xpu
