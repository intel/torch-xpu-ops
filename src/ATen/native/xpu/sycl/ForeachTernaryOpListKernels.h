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
