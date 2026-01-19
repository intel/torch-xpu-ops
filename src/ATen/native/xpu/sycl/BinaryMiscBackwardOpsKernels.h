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

TORCH_XPU_API void sigmoid_backward_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void tanh_backward_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void logit_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& eps_scalar);

} // namespace at::native::xpu
