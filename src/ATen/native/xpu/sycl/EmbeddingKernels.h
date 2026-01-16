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

TORCH_XPU_API Tensor embedding_dense_backward_kernel(
    const Tensor& grad_,
    const Tensor& indices_,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq);

TORCH_XPU_API Tensor& embedding_renorm_kernel(
    Tensor& self,
    const Tensor& indices,
    double max_norm,
    double norm_type);

} // namespace at::native::xpu
