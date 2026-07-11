/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Dispatch layer for stateless Philox key operations.
// Ported from CUDA: aten/src/ATen/native/cuda/PhiloxKeySplit.cu
// See PyTorch PR #177229.

#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/PhiloxKeySplitKernels.h>

namespace at::native {

Tensor _philox_key_split_xpu(const Tensor& key, int64_t num_splits) {
  return xpu::_philox_key_split_xpu(key, num_splits);
}

Tensor _philox_key_fold_in_xpu(const Tensor& key, int64_t data) {
  return xpu::_philox_key_fold_in_xpu(key, data);
}

} // namespace at::native
