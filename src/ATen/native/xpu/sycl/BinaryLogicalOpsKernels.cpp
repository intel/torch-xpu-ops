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
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/BinaryOps.h>
#include <ATen/native/xpu/sycl/BinaryInternal.h>
#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/BinaryLogicalOpsKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LogicalAndFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return a && b;
  }
};

void logical_and_kernel(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_and_xpu", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
          iter, LogicalAndFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(
        kHalf, kBool, ScalarType::BFloat16, dtype, "logical_and_xpu", [&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
              iter, LogicalAndFunctor<scalar_t>());
        });
  }
}

template <typename scalar_t>
struct LogicalOrFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return a || b;
  }
};

void logical_or_kernel(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_or_xpu", [&]() {
      gpu_kernel_with_scalars(iter, LogicalOrFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(
        kHalf, kBool, ScalarType::BFloat16, dtype, "logical_or_xpu", [&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
              iter, LogicalOrFunctor<scalar_t>());
        });
  }
}

template <typename scalar_t>
struct LogicalXorFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return bool(a) != bool(b);
  }
};

void logical_xor_kernel(TensorIterator& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logical_xor_xpu", [&]() {
      gpu_kernel_with_scalars(iter, LogicalXorFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(
        kHalf, kBool, ScalarType::BFloat16, dtype, "logical_xor_xpu", [&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
              iter, LogicalXorFunctor<scalar_t>());
        });
  }
}

} // namespace at::native::xpu
