/*
 * Copyright 2025 Intel Corporation
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

bool is_scaled_grouped_mm_available();

void f8f8bf16_scaled_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor scale_a,
    at::Tensor scale_b,
    std::optional<at::Tensor> offs,
    at::Tensor& out);

} // namespace at::native::xpu
