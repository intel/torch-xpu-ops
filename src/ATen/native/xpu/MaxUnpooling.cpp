/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/xpu/sycl/MaxUnpoolingKernels.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

Tensor& max_unpooling2d_forward_out_xpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    Tensor& out) {
  native::xpu::max_unpooling2d_forward_kernel(out, self, indices, output_size);
  return out;
}

Tensor max_unpooling2d_forward_xpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto out = at::empty({0}, self.options());
  max_unpooling2d_forward_out_xpu(self, indices, output_size, out);
  return out;
}

Tensor& max_unpooling3d_forward_out_xpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& out) {
  native::xpu::max_unpooling3d_forward_kernel(
      out, self, indices, output_size, stride, padding);
  return out;
}

Tensor max_unpooling3d_forward_xpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto out = at::empty({0}, self.options());
  max_unpooling3d_forward_out_xpu(
      self, indices, output_size, stride, padding, out);
  return out;
}

} // namespace at::native
