/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Dispatch layer for stateless Philox distribution operations.
// Ported from CUDA: aten/src/ATen/native/cuda/PhiloxDistribution.cu
// See PyTorch PR #177230.

#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/PhiloxDistributionKernels.h>

namespace at::native {

Tensor& _philox_uniform_xpu_(
    Tensor& self,
    const Tensor& key,
    double low,
    double high) {
  return xpu::_philox_uniform_xpu_(self, key, low, high);
}

Tensor& _philox_normal_xpu_(
    Tensor& self,
    const Tensor& key,
    double mean,
    double stddev) {
  return xpu::_philox_normal_xpu_(self, key, mean, stddev);
}

} // namespace at::native
