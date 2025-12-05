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

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void convert_indices_from_coo_to_csr_structured_kernel(
    const Tensor& input,
    const int64_t size,
    const bool out_int32,
    const Tensor& result);

TORCH_XPU_API void convert_indices_from_csr_to_coo_structured_kernel(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool out_int32,
    const bool transpose,
    const Tensor& result);

TORCH_XPU_API Tensor _sparse_csr_sum_xpu_kernel(
    const Tensor& input,
    IntArrayRef dims_to_sum,
    bool keepdim,
    std::optional<ScalarType> dtype);

TORCH_XPU_API Tensor _sparse_csr_prod_xpu_kernel(
    const Tensor& input,
    IntArrayRef dims_to_reduce,
    bool keepdim,
    std::optional<ScalarType> dtype);

} // namespace at::native::xpu
