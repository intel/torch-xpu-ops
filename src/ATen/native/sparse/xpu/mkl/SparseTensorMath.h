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

namespace at::native::xpu {

TORCH_XPU_API Tensor _sspaddmm_mkl_out(
    const Tensor& row_indices,
    const Tensor& col_indices,
    const Tensor& values1,
    const Tensor& mat2,
    const Tensor& self,
    const Scalar& beta,
    const Scalar& alpha,
    int64_t dim_i,
    int64_t dim_j,
    int64_t dim_k,
    int64_t nnz1);

} // namespace at::native::xpu
