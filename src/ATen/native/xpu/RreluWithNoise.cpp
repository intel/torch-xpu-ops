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
#include <ATen/native/xpu/sycl/RreluWithNoiseKernels.h>

namespace at {
namespace native {

Tensor& rrelu_with_noise_out_xpu(
    const Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator,
    Tensor& output) {
  return xpu::rrelu_with_noise_kernel(
      self, noise, lower, upper, training, generator, output);
}

Tensor rrelu_with_noise_xpu(
    const Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  Tensor output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::native::rrelu_with_noise_out_xpu(
      self, noise, lower, upper, training, generator, output);
}

Tensor& rrelu_with_noise_xpu_(
    Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  return at::native::rrelu_with_noise_out_xpu(
      self, noise, lower, upper, training, generator, self);
}

} // namespace native
} // namespace at
