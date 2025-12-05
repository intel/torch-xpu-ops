/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/FunctionOfAMatrixUtils.h>

#include <ATen/native/DispatchStub.h>

#include <ATen/native/xpu/sycl/FunctionOfAMatrixUtilsKernels.h>

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(
    _compute_linear_combination_stub,
    &xpu::_compute_linear_combination_kernel);

}
} // namespace at
