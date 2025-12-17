/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/xpu/sycl/ReduceMaxValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceMinValuesKernels.h>
#include <ATen/native/xpu/sycl/TensorCompareKernels.h>
#include <ATen/native/xpu/sycl/TensorModeKernel.h>
#include <comm/ReduceOpsUtils.h>

#include <ATen/ops/result_type_native.h>

namespace at {

namespace native {
namespace xpu {
void min_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto iter = meta::make_reduction(
      self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  native::xpu::min_kernel(iter);
}

void max_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto iter = meta::make_reduction(
      self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  native::xpu::max_kernel(iter);
}

template <class Stub>
void minmax_out_impl(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const Tensor& values,
    const Tensor& indices,
    Stub& stub) {
  NoNamesGuard guard;
  if (self.numel() > 0) {
    if (self.numel() == 1 && self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else {
      stub(values, indices, self, dim, keepdim);
    }
  }
}
} // namespace xpu

REGISTER_XPU_DISPATCH(where_kernel, &xpu::where_kernel);
REGISTER_XPU_DISPATCH(isposinf_stub, &xpu::isposinf_kernel);
REGISTER_XPU_DISPATCH(isneginf_stub, &xpu::isneginf_kernel);
REGISTER_XPU_DISPATCH(mode_stub, &xpu::mode_kernel);
REGISTER_XPU_DISPATCH(clamp_min_scalar_stub, &xpu::clamp_min_scalar_kernel);
REGISTER_XPU_DISPATCH(clamp_max_scalar_stub, &xpu::clamp_max_scalar_kernel);
REGISTER_XPU_DISPATCH(clamp_scalar_stub, &xpu::clamp_scalar_kernel);
REGISTER_XPU_DISPATCH(clamp_stub, &xpu::clamp_kernel);
REGISTER_XPU_DISPATCH(max_stub, &xpu::max_kernel_impl);
REGISTER_XPU_DISPATCH(min_stub, &xpu::min_kernel_impl)
REGISTER_XPU_DISPATCH(isin_default_stub, &xpu::isin_kernel);

void _assert_async_msg_xpu(
    const Tensor& self_tensor,
    std::string_view assert_msg) {
  xpu::_assert_async_msg_kernel(self_tensor, assert_msg);
}

void _assert_async_xpu(const Tensor& self_tensor) {
  _assert_async_msg_xpu(self_tensor, "");
}

} // namespace native
} // namespace at
