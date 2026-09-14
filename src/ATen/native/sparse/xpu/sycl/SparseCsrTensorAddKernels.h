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

#include <ATen/native/SparseTensorUtils.h>

namespace at::native::xpu {

TORCH_XPU_API void add_out_sparse_csr_kernel(
    const Tensor& A,
    const Tensor& B,
    const Scalar& alpha,
    Tensor& out);

} // namespace at::native::xpu
