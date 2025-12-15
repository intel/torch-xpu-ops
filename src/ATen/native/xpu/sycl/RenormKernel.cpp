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

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/RenormKernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct RenormScalarFactorFunctor {
  scalar_t operator()(scalar_t norm) const {
    const auto eps = static_cast<scalar_t>(1e-7);
    const auto one = static_cast<scalar_t>(1.0);
    return (norm > maxnorm_elm) ? maxnorm_elm / (norm + eps) : one;
  }

  RenormScalarFactorFunctor(scalar_t maxnorm_elm) : maxnorm_elm(maxnorm_elm) {}

 private:
  scalar_t maxnorm_elm;
};

void renorm_scale_factor_kernel(TensorIteratorBase& iter, double maxnorm) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "renorm_scale_factor_xpu",
      [&] {
        RenormScalarFactorFunctor<scalar_t> f(maxnorm);
        gpu_kernel(iter, f);
      });
}

} // namespace at::native::xpu
