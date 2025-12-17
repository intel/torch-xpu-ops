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

namespace at::native::xpu {

TORCH_XPU_API Tensor& binary_cross_entropy_kernel(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& loss);

TORCH_XPU_API Tensor& binary_cross_entropy_backward_kernel(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& grad_input);

} // namespace at::native::xpu
