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
#include <c10/core/ScalarType.h>

namespace at::native::xpu {

TORCH_XPU_API void norm_kernel(TensorIterator& iter, const Scalar& val);

} // namespace at::native::xpu
