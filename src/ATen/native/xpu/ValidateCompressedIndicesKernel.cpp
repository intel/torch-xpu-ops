/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/ValidateCompressedIndicesCommon.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native {

namespace {

template <typename func_t>
struct XPUKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    std::cout << "call xpu::gpu_kernel(iter, f)" << std::endl;
    xpu::gpu_kernel(iter, f);
  }
};

}

void _validate_compressed_sparse_indices_xpu(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  std::cout << "call _validate_compressed_sparse_indices_xpu" << std::endl;
  validate_compressed_sparse_indices_kernel<XPUKernelLauncher>(
      is_crow, cidx, idx, cdim, dim, nnz);
}

} // namespace at::native
