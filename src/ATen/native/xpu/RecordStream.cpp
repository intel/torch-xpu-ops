/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/xpu/XPUCachingAllocator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/record_stream_native.h>
#endif

namespace at::native {
void record_stream_xpu(Tensor& self, c10::Stream stream) {
  struct c10::StreamData3 data = stream.pack3();
  c10::xpu::XPUCachingAllocator::recordStream(
      self.storage().data_ptr(),
      at::xpu::XPUStream::unpack3(
          data.stream_id, data.device_index, data.device_type));
}
} // namespace at::native
