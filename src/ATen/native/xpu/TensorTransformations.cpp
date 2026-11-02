/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/TensorTransformations.h>
#include <ATen/native/xpu/sycl/TensorTransformationsKernels.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/roll_native.h>

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(flip_stub, &xpu::flip_kernel);

Tensor roll_xpu(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.empty() || shifts.size() != dims.size()) {
    // Flattened roll and argument errors.
    return at::native::roll_common(self, shifts, dims);
  }
  auto out = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (out.numel() == 0 || xpu::roll_kernel(self, out, shifts, dims)) {
    return out;
  }
  return at::native::roll(self, shifts, dims);
}

} // namespace native
} // namespace at
