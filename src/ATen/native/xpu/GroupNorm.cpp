/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/group_norm.h>
#include <ATen/native/xpu/sycl/GroupNormKernels.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(GroupNormKernel, &xpu::group_norm_kernel);
REGISTER_XPU_DISPATCH(
    GroupNormBackwardKernel,
    &xpu::group_norm_backward_kernel);
} // namespace native
} // namespace at
