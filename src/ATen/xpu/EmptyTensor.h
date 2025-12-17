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
#include <ATen/core/TensorBase.h>

namespace at::detail {

TORCH_XPU_API TensorBase empty_xpu(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<Device> device_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

TORCH_XPU_API TensorBase empty_xpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

TORCH_XPU_API TensorBase
empty_xpu(IntArrayRef size, const TensorOptions& options);

TORCH_XPU_API TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<Device> device_opt);

TORCH_XPU_API TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt);

TORCH_XPU_API TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options);

} // namespace at::detail
