/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void foreach_lerp_scalar_kernel(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight,
    TensorList result);

TORCH_XPU_API void foreach_lerp_scalar_kernel_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight);

} // namespace at::native::xpu
