/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Lerp.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/LerpKernels.h>

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(lerp_kernel_tensor_weight, &xpu::lerp_tensor_kernel);
REGISTER_XPU_DISPATCH(lerp_kernel_scalar_weight, &xpu::lerp_scalar_kernel);

} // namespace native

} // namespace at
