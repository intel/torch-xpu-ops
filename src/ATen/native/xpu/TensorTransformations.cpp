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
#include <c10/util/irange.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/constant_pad_nd_native.h>
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

// at::native::constant_pad_nd fills the output and then copies the input
// into a strided view of it (two passes); here the output is written once.
Tensor constant_pad_nd_xpu(
    const Tensor& self,
    IntArrayRef pad,
    const Scalar& value) {
  const int64_t ndim = self.dim();
  if (self.is_quantized() || pad.size() % 2 != 0 ||
      static_cast<int64_t>(pad.size() / 2) > ndim) {
    return at::native::constant_pad_nd(self, pad, value);
  }
  // Narrow away the negative pads as the composite does; the positive left
  // pads are the offsets of the input in the output.
  auto input = self;
  auto out_sizes = self.sizes().vec();
  c10::SmallVector<int64_t, 8> lo(ndim, 0);
  bool crop_only = true;
  for (const auto d : c10::irange(ndim - pad.size() / 2, ndim)) {
    const int64_t l = pad[2 * (ndim - d - 1)];
    const int64_t r = pad[2 * (ndim - d - 1) + 1];
    out_sizes[d] += l + r;
    if (out_sizes[d] < 0) {
      return at::native::constant_pad_nd(self, pad, value);
    }
    crop_only &= l <= 0 && r <= 0;
    if (l < 0) {
      input = input.narrow(d, -l, input.size(d) + l);
    } else {
      lo[d] = l;
    }
    if (r < 0) {
      input = input.narrow(d, 0, input.size(d) + r);
    }
  }
  if (crop_only) {
    // The composite returns input.clone(); a strided crop is gathered here.
    if (!input.is_non_overlapping_and_dense() && input.numel() > 0) {
      auto output = at::empty_like(input);
      if (xpu::constant_pad_nd_kernel(input, output, lo, value)) {
        return output;
      }
    }
    return input.clone();
  }
  auto output = at::empty(
      out_sizes, self.options().memory_format(self.suggest_memory_format()));
  if (output.numel() == 0 ||
      (input.numel() > 0 &&
       xpu::constant_pad_nd_kernel(input, output, lo, value))) {
    return output;
  }
  return at::native::constant_pad_nd(self, pad, value);
}

} // namespace native
} // namespace at
