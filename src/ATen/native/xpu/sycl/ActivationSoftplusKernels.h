/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void softplus_kernel(
    TensorIteratorBase& iter,
    const Scalar& beta_,
    const Scalar& threshold_);

TORCH_XPU_API void softplus_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& beta_,
    const Scalar& threshold_);

} // namespace at::native::xpu
