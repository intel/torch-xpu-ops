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

bool is_grouped_mm_available();

void bf16bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias,
    at::Tensor& out);

} // namespace at::native::xpu
