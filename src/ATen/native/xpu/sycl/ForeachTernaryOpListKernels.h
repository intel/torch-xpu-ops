/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void foreach_lerp_list_kernel(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3,
    TensorList result);

TORCH_XPU_API void foreach_lerp_list_kernel_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3);

} // namespace at::native::xpu
