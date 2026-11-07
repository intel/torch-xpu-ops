/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/CompareKernels.h>

#include <ATen/ops/isinf_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/isinf_native.h>
#include <c10/core/DeviceGuard.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(eq_stub, &xpu::eq_kernel);
REGISTER_XPU_DISPATCH(ne_stub, &xpu::ne_kernel);
REGISTER_XPU_DISPATCH(le_stub, &xpu::le_kernel);
REGISTER_XPU_DISPATCH(lt_stub, &xpu::lt_kernel);
REGISTER_XPU_DISPATCH(ge_stub, &xpu::ge_kernel);
REGISTER_XPU_DISPATCH(gt_stub, &xpu::gt_kernel);

namespace {

// The composite isinf runs abs() and a comparison as two kernels.
// Floating inputs use one kernel.
bool isinf_uses_kernel(const Tensor& self) {
  const auto dtype = self.scalar_type();
  return self.is_xpu() &&
      (dtype == kHalf || dtype == kBFloat16 || dtype == kFloat ||
       dtype == kDouble);
}

} // namespace

Tensor isinf_xpu(const Tensor& self) {
  if (!isinf_uses_kernel(self)) {
    return at::compositeexplicitautograd::isinf(self);
  }
  c10::OptionalDeviceGuard guard(device_of(self));
  auto iter = TensorIteratorConfig()
                  .add_owned_output(Tensor())
                  .add_const_input(self)
                  .check_all_same_dtype(false)
                  .declare_static_dtype_and_device(kBool, self.device())
                  .build();
  xpu::isinf_kernel(iter);
  return iter.output();
}

} // namespace native
} // namespace at
