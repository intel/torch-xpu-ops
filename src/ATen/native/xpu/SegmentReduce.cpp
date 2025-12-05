/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/native/DispatchStub.h>
#include <ATen/native/SegmentReduce.h>

#include <ATen/native/xpu/sycl/SegmentReduceKernels.h>

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(
    _segment_reduce_lengths_stub,
    &xpu::_segment_reduce_lengths_kernel);
REGISTER_XPU_DISPATCH(
    _segment_reduce_offsets_stub,
    &xpu::_segment_reduce_offsets_kernel);
REGISTER_XPU_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &xpu::_segment_reduce_lengths_backward_kernel);
REGISTER_XPU_DISPATCH(
    _segment_reduce_offsets_backward_stub,
    &xpu::_segment_reduce_offsets_backward_kernel);

} // namespace native
} // namespace at
