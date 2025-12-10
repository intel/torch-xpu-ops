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
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/xpu/sycl/AiryAiKernel.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(special_airy_ai_stub, &xpu::airy_ai_kernel);

} // namespace native
} // namespace at