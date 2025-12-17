/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from Torchvision
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/NMSKernel.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>

namespace at::native::xpu {

Tensor nms(const Tensor& dets, const Tensor& scores, double iou_threshold_) {
  float iou_threshold = (float)iou_threshold_;
  TORCH_CHECK(dets.is_xpu(), "dets must be a XPU tensor");
  TORCH_CHECK(scores.is_xpu(), "scores must be a XPU tensor");

  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0))

  c10::DeviceGuard device_guard(dets.device());

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  auto order_t = std::get<1>(
      scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));
  auto dets_sorted = dets.index_select(0, order_t).contiguous();

  auto keep = nms_kernel(dets_sorted, iou_threshold);
  return order_t.masked_select(keep);
}

} // namespace at::native::xpu
