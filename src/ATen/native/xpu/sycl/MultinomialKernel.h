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
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void multinomial_kernel(
    Tensor& result,
    const Tensor& self,
    const int64_t n_sample,
    std::optional<Generator> generator);

} // namespace at::native::xpu
