/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/ReduceMaxValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceMinValuesKernels.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>

namespace at {

void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = native::make_reduction(
      "min_all", result, input, IntArrayRef{}, false, dtype);
  native::xpu::min_all_kernel(iter);
}

void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = native::make_reduction(
      "max_all", result, input, IntArrayRef{}, false, dtype);
  native::xpu::max_all_kernel(iter);
}

namespace native {
REGISTER_XPU_DISPATCH(min_all_stub, &min_all_kernel_impl);
REGISTER_XPU_DISPATCH(max_all_stub, &max_all_kernel_impl);
} // namespace native

} // namespace at
