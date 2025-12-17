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
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(addcmul_stub, &xpu::addcmul_kernel);
REGISTER_XPU_DISPATCH(addcdiv_stub, &xpu::addcdiv_kernel);
} // namespace native
} // namespace at
