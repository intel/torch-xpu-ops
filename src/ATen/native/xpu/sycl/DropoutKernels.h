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

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor> dropout_kernel(
    const Tensor& self,
    double p,
    std::optional<bool> train);

TORCH_XPU_API Tensor
dropout_backward_kernel(const Tensor& grad, const Tensor& mask, double scale);

TORCH_XPU_API std::tuple<Tensor, Tensor> fused_dropout_kernel(
    const Tensor& self,
    double p,
    std::optional<Generator> gen_);

TORCH_XPU_API Tensor
masked_scale_kernel(const Tensor& self, const Tensor& mask, double scale);

} // namespace xpu
} // namespace native
} // namespace at
