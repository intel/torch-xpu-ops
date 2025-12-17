/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/DispatchStub.h>
#include <ATen/native/Distance.h>
#include <ATen/native/xpu/sycl/DistanceKernels.h>

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(cdist_stub, &xpu::cdist_kernel);
REGISTER_XPU_DISPATCH(cdist_backward_stub, &xpu::cdist_backward_kernel);
REGISTER_XPU_DISPATCH(pdist_forward_stub, &xpu::pdist_forward_kernel);
REGISTER_XPU_DISPATCH(pdist_backward_stub, &xpu::pdist_backward_kernel);

} // namespace native
} // namespace at
