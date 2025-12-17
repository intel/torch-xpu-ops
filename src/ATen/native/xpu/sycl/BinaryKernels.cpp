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
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/BinaryInternal.h>
#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/BinaryKernels.h>

namespace at::native::xpu {

template <typename opmath_t>
struct AddFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a + alpha_ * b;
  }
  AddFunctor(opmath_t alpha) : alpha_(alpha) {}

 private:
  opmath_t alpha_;
};

void add_kernel(TensorIteratorBase& iter, const c10::Scalar& alpha) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(
        iter, AddFunctor(alpha.to<opmath_t>()));
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "add_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_gpu_kernel_with_scalars<scalar_t>(
              iter, AddFunctor(alpha.to<opmath_t>()));
        });
  }
}

void sub_kernel(TensorIteratorBase& iter, const c10::Scalar& alpha) {
  add_kernel(iter, -alpha);
}

void mul_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, MulFunctor<opmath_t>());
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MulFunctor<opmath_t>());
        });
  }
}

} // namespace at::native::xpu
