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
#include <ATen/native/TransposeType.h>

namespace at::native::xpu {

TORCH_XPU_API void linalg_cholesky_ex_kernel(
    const Tensor& A,
    bool upper,
    bool check_errors,
    const Tensor& L,
    const Tensor& info);

TORCH_XPU_API void lu_solve_mkl(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType trans);

TORCH_XPU_API void lu_factor_mkl(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& info,
    bool pivot);

TORCH_XPU_API void triangular_solve_mkl(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular);

} // namespace at::native::xpu
