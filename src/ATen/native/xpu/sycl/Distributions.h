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
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

TORCH_XPU_API void launch_poisson_kernel(
    const TensorBase& ret,
    const TensorBase& lambda,
    XPUGeneratorImpl* gen);

TORCH_XPU_API void launch_binomial_kernel(
    TensorIteratorBase& iter,
    XPUGeneratorImpl* gen);

TORCH_XPU_API void launch_gamma_kernel(
    Tensor& ret,
    const Tensor& alpha,
    XPUGeneratorImpl* gen);

TORCH_XPU_API void launch_standard_gamma_grad_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void launch_dirichlet_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void launch_dirichlet_grad_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
