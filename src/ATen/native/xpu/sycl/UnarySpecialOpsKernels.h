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

namespace at::native::xpu {

TORCH_XPU_API void sigmoid_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void erf_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void erfc_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void erfinv_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void exp2_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void logit_kernel(
    TensorIteratorBase& iter,
    const Scalar& eps_scalar);

TORCH_XPU_API void i0_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void i0e_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void i1_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void i1e_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void ndtri_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void log_ndtr_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void entr_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void erfcx_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void sinc_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void kaiser_window_kernel(
    TensorIteratorBase& iter,
    int64_t window_length,
    double beta);

} // namespace at::native::xpu
