/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/ScalarOps.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>

#include <ATen/native/DispatchStub.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/ScanKernels.h>
#include <ATen/native/xpu/sycl/ReduceMaxValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceMinValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceOpsKernels.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>
#include <torch/library.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(sum_stub, &xpu::sum_kernel);
REGISTER_XPU_DISPATCH(mean_stub, &xpu::mean_kernel);
REGISTER_XPU_DISPATCH(prod_stub, &xpu::prod_kernel);
REGISTER_XPU_DISPATCH(argmax_stub, &xpu::argmax_kernel);
REGISTER_XPU_DISPATCH(argmin_stub, &xpu::argmin_kernel);
REGISTER_XPU_DISPATCH(and_stub, &xpu::and_kernel);
REGISTER_XPU_DISPATCH(or_stub, &xpu::or_kernel);
REGISTER_XPU_DISPATCH(max_values_stub, &xpu::max_values_kernel);
REGISTER_XPU_DISPATCH(min_values_stub, &xpu::min_values_kernel);
REGISTER_XPU_DISPATCH(std_var_stub, &xpu::std_var_kernel);
REGISTER_XPU_DISPATCH(cumsum_stub, &xpu::cumsum_kernel);
REGISTER_XPU_DISPATCH(cumprod_stub, &xpu::cumprod_kernel);
REGISTER_XPU_DISPATCH(nansum_stub, &xpu::nansum_kernel);

void cummax_helper_xpu(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  at::native::xpu::cummax_kernel(self, values, indices, dim);
}

void cummin_helper_xpu(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  at::native::xpu::cummin_kernel(self, values, indices, dim);
}

Tensor& _logcumsumexp_out_xpu(const Tensor& self, int64_t dim, Tensor& result) {
  return at::native::xpu::logcumsumexp_kernel(self, dim, result);
}

Tensor _logcumsumexp_xpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _logcumsumexp_out_xpu(self, dim, result);
}

void aminmax_impl(
    const Tensor& self,
    int64_t dim_opt,
    bool keepdim,
    Tensor& min,
    Tensor& max) {
  auto dtype = self.scalar_type();
  TensorIterator iter =
      make_reduction("aminmax_xpu", min, max, self, dim_opt, keepdim, dtype);
  if (iter.numel() != 0) {
    native::xpu::aminmax_kernel(iter);
  }
}

void aminmax_allreduce_impl(const Tensor& self, Tensor& min, Tensor& max) {
  auto dtype = self.scalar_type();
  auto iter = make_reduction(
      "aminmax_xpu", min, max, self, IntArrayRef{}, false, dtype);
  TORCH_CHECK(
      iter.numel() > 0, "min_max on a tensor with no elements is not defined.");
  native::xpu::aminmax_allreduce_kernel(iter);
}

REGISTER_XPU_DISPATCH(aminmax_stub, &aminmax_impl);
REGISTER_XPU_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_impl)

} // namespace native
} // namespace at
