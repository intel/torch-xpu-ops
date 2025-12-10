/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
