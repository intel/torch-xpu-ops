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
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/FillKernel.h>
namespace at::native {
REGISTER_XPU_DISPATCH(fill_stub, &native::xpu::fill_kernel);
} // namespace at::native
