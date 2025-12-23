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

TORCH_XPU_API std::vector<Tensor> foreach_norm_kernel(
    TensorList tensors,
    const Scalar& ord,
    double p,
    std::optional<ScalarType> dtype);

TORCH_XPU_API std::vector<Tensor> foreach_max_kernel(TensorList tensors);

} // namespace at::native::xpu
