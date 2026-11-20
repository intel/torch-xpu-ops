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

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void copy_kernel(TensorIteratorBase& iter);

// Copies raw bytes between two device-accessible USM pointers. Both must be
// reachable from the current device, i.e. device memory or host memory
// allocated through SYCL (pinned memory).
TORCH_XPU_API void copy_bytes_kernel(void* dst, const void* src, size_t nbytes);

} // namespace at::native::xpu
