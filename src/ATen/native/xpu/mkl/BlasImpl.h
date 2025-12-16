/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

Tensor& mm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out);

Tensor& bmm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out);

Tensor& addmm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out);

Tensor& baddbmm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out);

} // namespace at::native::xpu
