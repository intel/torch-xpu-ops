/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/nested/NestedTensorBinaryOps.h>
#include <ATen/native/nested/xpu/sycl/NestedTensorBinaryOpsKernels.h>

namespace at::native {

REGISTER_XPU_DISPATCH(
    nested_dense_elementwise_stub,
    &xpu::_nested_op_dense_esuhm_xpu)

} // namespace at::native
