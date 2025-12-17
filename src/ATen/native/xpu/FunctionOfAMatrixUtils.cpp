/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
