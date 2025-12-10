/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
