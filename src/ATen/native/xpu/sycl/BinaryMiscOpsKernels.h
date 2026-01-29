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

TORCH_XPU_API void mse_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void smooth_l1_kernel(TensorIteratorBase& iter, double beta);

TORCH_XPU_API void huber_kernel(TensorIterator& iter, double delta);

TORCH_XPU_API void xlogy_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void xlog1py_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void ldexp_kernel(TensorIteratorBase& iter);
} // namespace at::native::xpu
