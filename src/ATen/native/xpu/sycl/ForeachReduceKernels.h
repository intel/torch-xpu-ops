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
#include <comm/xpu_aten.h>

namespace at::native::xpu {

// True when the current device lists 32 among its sub-group sizes. The fused
// foreach reductions pin their stage-1 kernel to that width; other devices
// use the slow path.
TORCH_XPU_API bool foreach_reduce_supports_sub_group_size_32();

TORCH_XPU_API std::vector<Tensor> foreach_norm_kernel(
    TensorList tensors,
    const Scalar& ord,
    double p,
    std::optional<ScalarType> dtype);

TORCH_XPU_API std::vector<Tensor> foreach_powsum_kernel(
    TensorList tensors,
    const Scalar& ord,
    double p,
    std::optional<ScalarType> dtype);

TORCH_XPU_API std::vector<Tensor> foreach_max_kernel(TensorList tensors);

} // namespace at::native::xpu
