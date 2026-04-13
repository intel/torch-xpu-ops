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

TORCH_XPU_API void svd_mkl(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    const std::optional<std::string_view>& driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const Tensor& info);

TORCH_XPU_API void eigh_mkl(
    const Tensor& eigenvalues,
    const Tensor& eigenvectors,
    const Tensor& infos,
    bool upper,
    bool compute_eigenvectors);

} // namespace at::native::xpu
