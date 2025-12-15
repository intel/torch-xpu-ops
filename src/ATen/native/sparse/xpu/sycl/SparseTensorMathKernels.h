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

using namespace at::sparse;

TORCH_XPU_API SparseTensor& add_sparse_kernel(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_);

TORCH_XPU_API SparseTensor& mul_sparse_kernel(
    const Tensor& t_,
    const Tensor& src_,
    SparseTensor& r_);

TORCH_XPU_API Tensor _sparse_sum_backward_kernel(
    const Tensor& grad_,
    const SparseTensor& input_,
    IntArrayRef dims_to_sum);

} // namespace at::native::xpu
