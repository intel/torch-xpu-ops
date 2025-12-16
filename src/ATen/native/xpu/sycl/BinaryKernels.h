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

TORCH_XPU_API void add_kernel(TensorIteratorBase& iter, const Scalar& alpha);

TORCH_XPU_API void sub_kernel(TensorIteratorBase& iter, const Scalar& alpha);

TORCH_XPU_API void mul_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void div_true_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void div_trunc_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void div_floor_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
