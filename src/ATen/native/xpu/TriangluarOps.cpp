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
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/TriangularOpsKernels.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/tril_native.h>
#include <ATen/ops/triu_native.h>

namespace at::native {

TORCH_IMPL_FUNC(tril_xpu)(const Tensor& self, int64_t k, const Tensor& result) {
  if (self.numel() != 0) {
    xpu::tril_kernel(result, self, k);
  }
}

TORCH_IMPL_FUNC(triu_xpu)(const Tensor& self, int64_t k, const Tensor& result) {
  if (self.numel() != 0) {
    xpu::triu_kernel(result, self, k);
  }
}

Tensor trace_xpu(const Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "expected a matrix");
  return self.diagonal().sum();
}

} // namespace at::native