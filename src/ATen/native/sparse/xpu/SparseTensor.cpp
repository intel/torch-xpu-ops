/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/xpu/sycl/SparseTensorKernels.h>

namespace at::native {

using namespace at::sparse;

SparseTensor _coalesce_sparse_xpu(const SparseTensor& self) {
  return xpu::coalesce_sparse_kernel(self);
}

REGISTER_XPU_DISPATCH(flatten_indices_stub, &xpu::flatten_indices_kernel);

} // namespace at::native
