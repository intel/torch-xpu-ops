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

#include <ATen/NumericUtils.h>
#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/BinaryMiscOpsKernels.h>

namespace at::native::xpu {
template <typename scalar_t>
struct MSEFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    auto diff = a - b;
    return diff * diff;
  }
};

void mse_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mse_xpu",
      [&]() { gpu_kernel(iter, MSEFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct SmoothL1Functor {
  scalar_t operator()(scalar_t input, scalar_t target) const {
    auto z = std::abs(input - target);
    return z < beta_val ? scalar_t(0.5) * z * z / beta_val
                        : z - scalar_t(0.5) * beta_val;
  }
  SmoothL1Functor(scalar_t beta_val) : beta_val(beta_val) {}

 private:
  scalar_t beta_val;
};

void smooth_l1_kernel(TensorIteratorBase& iter, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "smooth_l1_xpu",
      [&iter, beta]() {
        scalar_t beta_val(beta);
        SmoothL1Functor<scalar_t> f(beta_val);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t>
struct HuberFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    auto z = std::abs(a - b);
    return z < delta_val_ ? scalar_t(0.5) * z * z
                          : delta_val_ * (z - scalar_t(0.5) * delta_val_);
  }
  HuberFunctor(scalar_t delta_val) : delta_val_(delta_val) {}

 private:
  scalar_t delta_val_;
};

void huber_kernel(TensorIterator& iter, double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "huber_xpu", [&iter, delta] {
        scalar_t delta_val(delta);
        gpu_kernel(iter, HuberFunctor<scalar_t>(delta_val));
      });
}

template <typename scalar_t>
struct XlogyFunctor {
  scalar_t operator()(scalar_t x, scalar_t y) const {
    if (at::_isnan(y)) {
      return NAN;
    }
    if (x == 0) {
      return 0;
    }
    return x * std::log(y);
  }
};

void xlogy_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "xlogy_xpu",
      [&]() { gpu_kernel_with_scalars(iter, XlogyFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct Xlog1pyFunctor {
  scalar_t operator()(scalar_t x, scalar_t y) const {
    if (at::_isnan(y)) {
      return NAN;
    }
    if (x == 0) {
      return 0;
    }
    return x * std::log1p(y);
  }
};

void xlog1py_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "xlog1py_xpu",
      [&]() { gpu_kernel_with_scalars(iter, Xlog1pyFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
