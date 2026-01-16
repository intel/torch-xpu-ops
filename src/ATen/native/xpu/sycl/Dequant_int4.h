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
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void dequant_int4_kernel(
    const Tensor& weight_int4,
    Tensor& weight,
    int qGroupSize,
    const Tensor& qScaleAndZeros);

} // namespace at::native::xpu
