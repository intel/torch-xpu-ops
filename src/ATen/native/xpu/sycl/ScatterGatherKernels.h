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

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void gather_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index);

TORCH_XPU_API void scatter_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src);

TORCH_XPU_API void scatter_fill_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& src);

TORCH_XPU_API void scatter_add_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src);

TORCH_XPU_API void scatter_reduce_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce);

TORCH_XPU_API void scatter_reduce_two_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const ReductionType& reduce);

TORCH_XPU_API void scatter_scalar_reduce_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Scalar& value,
    const ReductionType& reduce);

} // namespace xpu
} // namespace native
} // namespace at
