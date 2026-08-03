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
  return at::TensorOptions()
      .dtype(c10::CppTypeToScalarType<T>::value)
      .device(kXPU)
      .memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
}
} // namespace at
