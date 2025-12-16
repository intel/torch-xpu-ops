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

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void bitwise_and_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void bitwise_or_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void bitwise_xor_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
