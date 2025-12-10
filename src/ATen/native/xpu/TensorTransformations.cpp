/*
 * Copyright 2020-2025 Intel Corporation
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

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(flip_stub, &xpu::flip_kernel);

Tensor roll_xpu(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.size() != 1 || shifts.size() != 1) {
    return at::native::roll_common(self, shifts, dims);
  }

  auto in_tensor = self;
  if (!self.is_contiguous()) {
    in_tensor = self.contiguous();
  }
  auto out_tensor = at::empty_like(in_tensor, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (out_tensor.numel() == 0) {
    return out_tensor;
  }

  xpu::roll_kernel(in_tensor, out_tensor, shifts, dims);

  return out_tensor;
}

} // namespace native
} // namespace at
