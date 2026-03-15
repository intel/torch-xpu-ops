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

#include <comm/SYCLContext.h>

namespace at {
template <typename T>
static inline TensorOptions map_options() {
  if (std::is_same_v<T, uint8_t>)
    return at::TensorOptions().dtype(kByte).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, int8_t>)
    return at::TensorOptions().dtype(kChar).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, int16_t>)
    return at::TensorOptions().dtype(kShort).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, int32_t>)
    return at::TensorOptions().dtype(kInt).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, int64_t>)
    return at::TensorOptions().dtype(kLong).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, float>)
    return at::TensorOptions().dtype(kFloat).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, double>)
    return at::TensorOptions().dtype(kDouble).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, at::Half>)
    return at::TensorOptions().dtype(kHalf).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, at::BFloat16>)
    return at::TensorOptions().dtype(kBFloat16).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, bool>)
    return at::TensorOptions().dtype(kBool).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, c10::complex<float>>)
    return at::TensorOptions()
        .dtype(kComplexFloat)
        .device(kXPU)
        .memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, c10::complex<double>>)
    return at::TensorOptions()
        .dtype(kComplexDouble)
        .device(kXPU)
        .memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, uint16_t>)
    return at::TensorOptions().dtype(kUInt16).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, uint32_t>)
    return at::TensorOptions().dtype(kUInt32).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else if (std::is_same_v<T, uint64_t>)
    return at::TensorOptions().dtype(kUInt64).device(kXPU).memory_format(
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  else {
    AT_ERROR("PSTLFunctions: data type cannot be mapped to tensor's dtype.");
  }
  return at::TensorOptions();
}
} // namespace at