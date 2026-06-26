/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/xpu/XPUGeneratorBridge.h>
#include <comm/DeviceProperties.h>
#include <comm/Runtime.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/DistributionKernels.h>

namespace at {
namespace native {
namespace xpu {

void bernoulli_tensor_kernel(
    const TensorBase& self,
    const TensorBase& p_,
    std::optional<Generator> gen_) {
  auto generator = get_generator_or_default<at::XPUGeneratorImpl>(
      gen_, at::Generator(c10::xpu::getDefaultXPUGeneratorBridge(-1)));
  at::native::templates::xpu::bernoulli_kernel(self, p_, generator);
}

void bernoulli_scalar_kernel(
    const TensorBase& self,
    double p,
    std::optional<Generator> gen) {
  auto iter = TensorIterator::borrowing_nullary_op(self);
  auto generator = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::Generator(c10::xpu::getDefaultXPUGeneratorBridge(-1)));
  at::native::templates::xpu::bernoulli_kernel(iter, p, generator);
}

} // namespace xpu
} // namespace native
} // namespace at
