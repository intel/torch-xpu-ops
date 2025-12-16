/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionKernels.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

void cauchy_kernel(
    TensorIteratorBase& iter,
    double median,
    double sigma,
    std::optional<Generator> gen) {
  auto generator = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  at::native::templates::xpu::cauchy_kernel(iter, median, sigma, generator);
}

} // namespace at::native::xpu
