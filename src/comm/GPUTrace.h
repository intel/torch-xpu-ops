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

#include <c10/core/impl/GPUTrace.h>
#include <c10/xpu/XPUStream.h>

namespace xpu {
namespace sycl {

// A blocking memcpy drains the current stream as far as the caller can observe,
// so report it to the GPU tracer the way c10::cuda::memcpy_and_sync does on
// CUDA. The reported handle is the current stream's sycl::queue pointer, which
// is what XPUStream::synchronize() reports and what Stream.native_handle
// exposes to Python.
inline void traceStreamSynchronization() {
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_synchronization(
        c10::kXPU,
        reinterpret_cast<uintptr_t>(&c10::xpu::getCurrentXPUStream().queue()));
  }
}

} // namespace sycl
} // namespace xpu
