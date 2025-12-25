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

#include <ATen/core/Generator.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void random_from_to_kernel(
    TensorIteratorBase& iter,
    uint64_t range,
    int64_t base,
    std::optional<Generator> gen_);

TORCH_XPU_API void random_full_64_bits_range_kernel(
    TensorIteratorBase& iter,
    std::optional<Generator> gen_);

TORCH_XPU_API void random_kernel(
    TensorIteratorBase& iter,
    std::optional<Generator> gen_);

TORCH_XPU_API void uniform_kernel(
    TensorIteratorBase& iter,
    double from,
    double to,
    std::optional<Generator> gen);

TORCH_XPU_API void normal_kernel(
    const TensorBase& self,
    double mean,
    double std,
    std::optional<Generator> gen);

TORCH_XPU_API void bernoulli_tensor_kernel(
    const TensorBase& self,
    const TensorBase& p_,
    std::optional<Generator> gen_);

TORCH_XPU_API void bernoulli_scalar_kernel(
    const TensorBase& self,
    double p,
    std::optional<Generator> gen);

TORCH_XPU_API void exponential_kernel(
    TensorIteratorBase& iter,
    double lambda,
    std::optional<Generator> gen);

TORCH_XPU_API void log_normal_kernel(
    TensorIteratorBase& iter,
    double mean,
    double std,
    std::optional<Generator> gen);

TORCH_XPU_API void cauchy_kernel(
    TensorIteratorBase& iter,
    double median,
    double sigma,
    std::optional<Generator> gen);

TORCH_XPU_API void geometric_kernel(
    TensorIteratorBase& iter,
    double p_,
    std::optional<Generator> gen);

} // namespace at::native::xpu
