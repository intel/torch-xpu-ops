/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/ScanKernels.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>

namespace at::native::xpu {

void launch_cumsum_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      self.scalar_type(),
      "cumsum_xpu",
      [&]() {
        scalar_t init = 0;
        scan<INCLUSIVE_TYPE, const scalar_t, scalar_t>(
            result, self, dim, init, std::plus<scalar_t>());
      });
}

} // namespace at::native::xpu
