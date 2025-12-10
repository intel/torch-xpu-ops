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

namespace at::native::xpu {

TORCH_XPU_API void digamma_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void polygamma_kernel(TensorIteratorBase& iter, int64_t n);

TORCH_XPU_API void lgamma_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
