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

#include <ATen/core/Tensor.h>

namespace at::native {

TORCH_XPU_API Tensor& mm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out);

TORCH_XPU_API Tensor& bmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out);

TORCH_XPU_API Tensor& addmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out);

TORCH_XPU_API Tensor& baddbmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out);

} // namespace at::native
