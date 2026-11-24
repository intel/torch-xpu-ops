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

#include <c10/core/ScalarType.h>
#include <comm/Macros.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
TORCH_XPU_API void _copy_kernel(scalar_t* dst, scalar_t* src, int n);

extern template TORCH_XPU_API void _copy_kernel<bool>(bool*, bool*, int);
extern template TORCH_XPU_API void _copy_kernel<int8_t>(int8_t*, int8_t*, int);
extern template TORCH_XPU_API void _copy_kernel<uint8_t>(
    uint8_t*,
    uint8_t*,
    int);
extern template TORCH_XPU_API void _copy_kernel<short>(short*, short*, int);
extern template TORCH_XPU_API void _copy_kernel<int>(int*, int*, int);
#ifdef _WIN32
extern template TORCH_XPU_API void _copy_kernel<__int64>(
    __int64*,
    __int64*,
    int);
#else
extern template TORCH_XPU_API void _copy_kernel<int64_t>(
    int64_t*,
    int64_t*,
    int);
#endif
extern template TORCH_XPU_API void _copy_kernel<double>(double*, double*, int);
extern template TORCH_XPU_API void _copy_kernel<float>(float*, float*, int);
extern template TORCH_XPU_API void _copy_kernel<c10::BFloat16>(
    c10::BFloat16*,
    c10::BFloat16*,
    int);
extern template TORCH_XPU_API void _copy_kernel<c10::Half>(
    c10::Half*,
    c10::Half*,
    int);
extern template TORCH_XPU_API void _copy_kernel<c10::Float8_e5m2>(
    c10::Float8_e5m2*,
    c10::Float8_e5m2*,
    int);
extern template TORCH_XPU_API void _copy_kernel<c10::Float8_e4m3fn>(
    c10::Float8_e4m3fn*,
    c10::Float8_e4m3fn*,
    int);

} // namespace xpu
} // namespace native
} // namespace at
