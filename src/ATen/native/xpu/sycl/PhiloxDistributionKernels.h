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

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor& _philox_uniform_xpu_(
    Tensor& self,
    const Tensor& key,
    double low,
    double high);

TORCH_XPU_API Tensor& _philox_normal_xpu_(
    Tensor& self,
    const Tensor& key,
    double mean,
    double stddev);

} // namespace at::native::xpu
