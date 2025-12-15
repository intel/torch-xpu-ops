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

#include <ATen/native/xpu/sycl/Reduce.h>
#include <ATen/native/xpu/sycl/SharedReduceOps.h>
#include <ATen/ops/imag.h>

#include <ATen/native/xpu/sycl/ReduceNormKernel.h>

namespace at::native::xpu {

// This reduction accumulates results as the type `acc_t`. By default, when
// `scalar_t` is complex, `acc_t` is the downgraded real number type.
// Otherwise, `acc_t` and `scalar_t` are the same type.
template <
    typename scalar_t,
    typename acc_t = typename scalar_value_type<scalar_t>::type,
    typename out_t = typename scalar_value_type<scalar_t>::type>
void norm_kernel_xpu_impl(TensorIterator& iter, double p) {
  if (p == static_cast<double>(0)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NormZeroOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(1)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NormOneOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(2)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NormTwoOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, AbsMaxOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(-INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter,
        AbsMinOps<scalar_t, acc_t, out_t>(),
        std::numeric_limits<acc_t>::infinity());
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(
        iter, NormOps<scalar_t, acc_t, out_t>{acc_t(p)}, 0);
  }
}

void norm_launch_kernel(TensorIterator& iter, double ord) {
  if (iter.dtype(0) == kHalf) {
    return norm_kernel_xpu_impl<at::Half, float>(iter, ord);
  } else if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_xpu_impl<at::Half, float, float>(iter, ord);
  } else if (iter.dtype(0) == kBFloat16) {
    return norm_kernel_xpu_impl<at::BFloat16, float>(iter, ord);
  } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_xpu_impl<at::BFloat16, float, float>(iter, ord);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "norm_xpu", [&] {
    norm_kernel_xpu_impl<scalar_t>(iter, ord);
  });
}

void norm_kernel(TensorIterator& iter, const Scalar& val) {
  double p;
  if (val.isIntegral(false)) {
    p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
    p = val.to<double>();
  } else {
    TORCH_CHECK(
        false, "norm_kernel_xpu_impl expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((p < 0) ? INFINITY : 0);
    return;
  }

  norm_launch_kernel(iter, p);

  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }
}

} // namespace at::native::xpu
