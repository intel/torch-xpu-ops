#pragma once
#include <ATen/core/TensorBase.h>

namespace at::detail {

TORCH_XPU_API TensorBase empty_xpu(
    IntArrayRef size,
    ScalarType dtype,
    c10::optional<Device> device_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

TORCH_XPU_API TensorBase empty_xpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

TORCH_XPU_API TensorBase
empty_xpu(IntArrayRef size, const TensorOptions& options);

TORCH_XPU_API TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    c10::optional<Device> device_opt);

TORCH_XPU_API TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt);

TORCH_XPU_API TensorBase empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options);

} // namespace at::detail
