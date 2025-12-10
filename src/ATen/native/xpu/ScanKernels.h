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
#include <cstdint>

namespace at::native::xpu {

TORCH_XPU_API void cumsum_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

TORCH_XPU_API void cumprod_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

TORCH_XPU_API void cummax_kernel(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim);

TORCH_XPU_API void cummin_kernel(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim);

TORCH_XPU_API Tensor& logcumsumexp_kernel(
    const Tensor& self,
    int64_t dim,
    Tensor& result);

TORCH_XPU_API void launch_cumsum_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

TORCH_XPU_API void launch_cumprod_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

TORCH_XPU_API void launch_cummax_kernel(
    const Tensor& self,
    const Tensor& values,
    const Tensor& indices,
    int64_t dim);

TORCH_XPU_API void launch_cummin_kernel(
    const Tensor& self,
    const Tensor& values,
    const Tensor& indices,
    int64_t dim);

TORCH_XPU_API void launch_logcumsumexp_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

} // namespace at::native::xpu
