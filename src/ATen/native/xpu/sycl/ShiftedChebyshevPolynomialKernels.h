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

#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void shifted_chebyshev_polynomial_t_kernel(
    TensorIteratorBase& iterator);

TORCH_XPU_API void shifted_chebyshev_polynomial_u_kernel(
    TensorIteratorBase& iterator);

TORCH_XPU_API void shifted_chebyshev_polynomial_v_kernel(
    TensorIteratorBase& iterator);

TORCH_XPU_API void shifted_chebyshev_polynomial_w_kernel(
    TensorIteratorBase& iterator);

} // namespace at::native::xpu
