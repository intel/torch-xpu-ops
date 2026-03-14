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
#include <c10/core/ScalarType.h>

#include <optional>

namespace at::native {

TORCH_XPU_API Tensor _grouped_mm_xpu(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype);

} // namespace at::native
