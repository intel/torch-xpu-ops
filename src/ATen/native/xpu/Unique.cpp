/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/sycl/UniqueKernels.h>

namespace at {

namespace native {

std::tuple<Tensor, Tensor, Tensor> unique_dim_xpu(
    const Tensor& self,
    const int64_t dim,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  return xpu::unique_dim_kernel(self, dim, return_inverse, return_counts);
}

std::tuple<Tensor, Tensor> _unique_xpu(
    const Tensor& self,
    const bool sorted,
    const bool return_inverse) {
  return xpu::_unique_kernel(self, return_inverse);
}

std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_xpu(
    const at::Tensor& self,
    int64_t dim,
    bool return_inverse,
    bool return_counts) {
  return xpu::unique_dim_consecutive_kernel(
      self, dim, return_inverse, return_counts);
}

std::tuple<Tensor, Tensor, Tensor> unique_consecutive_xpu(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    std::optional<int64_t> dim) {
  if (!dim.has_value()) {
    return xpu::unique_consecutive_kernel(
        self, return_inverse, return_counts, dim);
  }
  return xpu::unique_dim_consecutive_kernel(
      self, dim.value(), return_inverse, return_counts);
}

std::tuple<Tensor, Tensor, Tensor> _unique2_xpu(
    const Tensor& self,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  return xpu::_unique2_kernel(self, return_inverse, return_counts);
}

} // namespace native

} // namespace at