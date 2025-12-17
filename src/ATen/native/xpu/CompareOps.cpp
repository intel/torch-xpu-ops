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
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/CompareKernels.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(eq_stub, &xpu::eq_kernel);
REGISTER_XPU_DISPATCH(ne_stub, &xpu::ne_kernel);
REGISTER_XPU_DISPATCH(le_stub, &xpu::le_kernel);
REGISTER_XPU_DISPATCH(lt_stub, &xpu::lt_kernel);
REGISTER_XPU_DISPATCH(ge_stub, &xpu::ge_kernel);
REGISTER_XPU_DISPATCH(gt_stub, &xpu::gt_kernel);
} // namespace native
} // namespace at
