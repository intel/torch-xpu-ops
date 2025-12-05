/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef USE_C10D_XCCL

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

namespace c10d {

void checkForNan(const at::Tensor& tensor, at::xpu::XPUStream& stream);

} // namespace c10d

#endif // USE_C10D_XCCL
