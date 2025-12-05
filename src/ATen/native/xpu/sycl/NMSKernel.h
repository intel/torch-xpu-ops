/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API Tensor nms_kernel(const Tensor& dets_sorted, float iou_threshold);

}
} // namespace native
} // namespace at
