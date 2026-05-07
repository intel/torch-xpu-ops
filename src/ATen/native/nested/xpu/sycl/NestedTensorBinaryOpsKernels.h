/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/native/nested/NestedTensorBinaryOps.h>
#include <torch/headeronly/macros/Export.h>

namespace at::native::xpu {

TORCH_XPU_API void _nested_op_dense_esuhm_xpu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const NESTED_DENSE_OP& op);

} // namespace at::native::xpu
