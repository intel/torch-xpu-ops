/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API TensorImpl* resize_impl_xpu_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool device_guard = true);

}
