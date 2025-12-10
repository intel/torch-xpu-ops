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
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnfoldBackward.h>
#include <ATen/native/xpu/sycl/UnfoldBackwardKernels.h>
#include <comm/xpu_aten.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(unfold_backward_stub, &xpu::unfold_backward_kernel);
}
} // namespace at
