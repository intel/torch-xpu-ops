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

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> unique_consecutive_kernel(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    std::optional<int64_t> dim);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_kernel(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse,
    const bool return_counts);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> unique_dim_kernel(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse,
    const bool return_counts);

TORCH_XPU_API std::tuple<Tensor, Tensor> _unique_kernel(
    const Tensor& self,
    const bool return_inverse);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> _unique2_kernel(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts);

} // namespace at::native::xpu
