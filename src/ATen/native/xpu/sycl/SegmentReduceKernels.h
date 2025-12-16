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

#include <ATen/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReductionType.h>
#include <optional>

namespace at::native::xpu {

TORCH_XPU_API Tensor _segment_reduce_lengths_kernel(
    native::ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const std::optional<Scalar>& initial);

TORCH_XPU_API Tensor _segment_reduce_offsets_kernel(
    native::ReductionType reduction,
    const Tensor& data,
    const Tensor& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial);

TORCH_XPU_API Tensor _segment_reduce_lengths_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    native::ReductionType reduction,
    const Tensor& lengths_contig,
    int64_t axis,
    const std::optional<Scalar>& initial);

TORCH_XPU_API Tensor _segment_reduce_offsets_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    native::ReductionType reduction,
    const Tensor& offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial);

} // namespace at::native::xpu
