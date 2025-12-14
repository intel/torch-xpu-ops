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

TORCH_XPU_API void addcmul_kernel(
    TensorIteratorBase& iter,
    const Scalar& value);

TORCH_XPU_API void addcdiv_kernel(
    TensorIteratorBase& iter,
    const Scalar& value);

TORCH_XPU_API void mse_backward_kernel(
    TensorIterator& iter,
    const Scalar& value);

TORCH_XPU_API void smooth_l1_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double beta);

TORCH_XPU_API void huber_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double delta);

} // namespace at::native::xpu
