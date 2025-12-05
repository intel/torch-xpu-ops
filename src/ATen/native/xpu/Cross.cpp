/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Cross.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/xpu/sycl/CrossKernel.h>
#include <comm/RegisterUtils.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(cross_stub, &xpu::linalg_cross_kernel);
} // namespace native
} // namespace at
