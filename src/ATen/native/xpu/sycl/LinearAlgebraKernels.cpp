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
#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/LinearAlgebraKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AddrKernelFunctorForBetaIsFalse {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return alpha_val_ && vec1_val && vec2_val;
  }
  AddrKernelFunctorForBetaIsFalse(scalar_t alpha_val) : alpha_val_(alpha_val) {}

 private:
  scalar_t alpha_val_;
};

template <typename scalar_t>
struct AddrKernelFunctorForBetaIsTrue {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return (beta_val_ && self_val) || (alpha_val_ && vec1_val && vec2_val);
  }

  AddrKernelFunctorForBetaIsTrue(scalar_t alpha_val, scalar_t beta_val)
      : alpha_val_(alpha_val), beta_val_(beta_val) {}

 private:
  scalar_t alpha_val_;
  scalar_t beta_val_;
};

template <typename scalar_t>
struct AddrKernelFunctorForBetaIsZero {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return alpha_val_ * vec1_val * vec2_val;
  }
  AddrKernelFunctorForBetaIsZero(scalar_t alpha_val) : alpha_val_(alpha_val) {}

 private:
  scalar_t alpha_val_;
};

template <typename scalar_t>
struct AddrKernelFunctorForBetaIsNotZero {
  scalar_t operator()(scalar_t self_val, scalar_t vec1_val, scalar_t vec2_val)
      const {
    return beta_val_ * self_val + alpha_val_ * vec1_val * vec2_val;
  }

  AddrKernelFunctorForBetaIsNotZero(scalar_t alpha_val, scalar_t beta_val)
      : alpha_val_(alpha_val), beta_val_(beta_val) {}

 private:
  scalar_t alpha_val_;
  scalar_t beta_val_;
};

void addr_kernel(
    TensorIterator& iter,
    const Scalar& beta,
    const Scalar& alpha) {
  if (iter.dtype() == at::ScalarType::Bool) {
    using scalar_t = bool;
    auto beta_val = beta.to<scalar_t>();
    auto alpha_val = alpha.to<scalar_t>();

    // when beta is false, values in self should be ignored,
    // nans and infs in self should not propagate.
    if (beta_val == false) {
      AddrKernelFunctorForBetaIsFalse<scalar_t> f(alpha_val);
      gpu_kernel(iter, f);
    } else {
      AddrKernelFunctorForBetaIsTrue<scalar_t> f(alpha_val, beta_val);
      gpu_kernel(iter, f);
    }
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kBFloat16, kHalf, iter.dtype(), "addr_xpu", [&] {
        auto beta_val = beta.to<scalar_t>();
        auto alpha_val = alpha.to<scalar_t>();

        scalar_t zero_val(0);
        // when beta==0, values in self should be ignored,
        // nans and infs in self should not propagate.
        if (beta_val == zero_val) {
          AddrKernelFunctorForBetaIsZero<scalar_t> f(alpha_val);
          gpu_kernel(iter, f);
        } else {
          AddrKernelFunctorForBetaIsNotZero<scalar_t> f(alpha_val, beta_val);
          gpu_kernel(iter, f);
        }
      });
}

} // namespace at::native::xpu
