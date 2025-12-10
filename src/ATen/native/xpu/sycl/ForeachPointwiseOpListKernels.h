/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarListKernels.h>

namespace at::native::xpu {

#define FOREACH_POINTWISE_OP_TENSOR_KERNEL(NAME) \
  FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(NAME)

#define FOREACH_POINTWISE_OP_TENSOR_INPLACE_KERNEL(NAME) \
  FOREACH_POINTWISE_OP_SCALARLIST_INPLACE_KERNEL(NAME)

TORCH_XPU_API FOREACH_POINTWISE_OP_TENSOR_KERNEL(addcmul);
TORCH_XPU_API FOREACH_POINTWISE_OP_TENSOR_INPLACE_KERNEL(addcmul);
TORCH_XPU_API FOREACH_POINTWISE_OP_TENSOR_KERNEL(addcdiv);
TORCH_XPU_API FOREACH_POINTWISE_OP_TENSOR_INPLACE_KERNEL(addcdiv);

} // namespace at::native::xpu
