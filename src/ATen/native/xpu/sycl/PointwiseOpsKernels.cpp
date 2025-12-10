/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AddcmulFunctor {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return static_cast<accscalar_t>(a) +
        alpha_ * static_cast<accscalar_t>(b) * static_cast<accscalar_t>(c);
  }

  AddcmulFunctor(accscalar_t alpha) : alpha_(alpha) {}

 private:
  accscalar_t alpha_;
};

template <typename scalar_t>
struct AddcmulComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return a + alpha_ * b * c;
  }

  AddcmulComplexFunctor(scalar_t alpha) : alpha_(alpha) {}

 private:
  scalar_t alpha_;
};

void addcmul_kernel(TensorIteratorBase& iter, const Scalar& value) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "addcmul_xpu", [&]() {
      auto alpha = value.to<scalar_t>();
      gpu_kernel(iter, AddcmulComplexFunctor<scalar_t>(alpha));
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "addcmul_xpu",
        [&]() {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          auto alpha = value.to<accscalar_t>();
          gpu_kernel(iter, AddcmulFunctor<scalar_t>(alpha));
        });
  }
}

template <typename scalar_t>
struct AddcdivFunctor {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return a + alpha_ * (b / static_cast<accscalar_t>(c));
  }

  AddcdivFunctor(accscalar_t alpha) : alpha_(alpha) {}

 private:
  accscalar_t alpha_;
};

template <typename scalar_t>
struct AddcdivComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return a + alpha_ * (b / c);
  }

  AddcdivComplexFunctor(scalar_t alpha) : alpha_(alpha) {}

 private:
  scalar_t alpha_;
};

void addcdiv_kernel(TensorIteratorBase& iter, const Scalar& value) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "addcdiv_xpu", [&]() {
      auto alpha = value.to<scalar_t>();
      AddcdivComplexFunctor<scalar_t> f(alpha);
      gpu_kernel(iter, f);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "addcdiv_xpu",
        [&]() {
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          auto alpha = value.to<accscalar_t>();
          AddcdivFunctor<scalar_t> f(alpha);
          gpu_kernel(iter, f);
        });
  }
}

template <typename scalar_t>
struct MSEBackwardFunctor {
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return alpha_ * (a - b) * c;
  }
  MSEBackwardFunctor(scalar_t alpha) : alpha_(alpha) {}

 private:
  scalar_t alpha_;
};

void mse_backward_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mse_backward_xpu",
      [&]() {
        auto alpha = value.to<scalar_t>();
        gpu_kernel(iter, MSEBackwardFunctor<scalar_t>(alpha));
      });
}

template <typename scalar_t>
struct SmoothL1BackwardFunctor {
  scalar_t operator()(scalar_t input, scalar_t target, scalar_t grad_output)
      const {
    const auto x = input - target;
    if (x < -beta_val)
      return -norm_val * grad_output;
    else if (x > beta_val)
      return norm_val * grad_output;
    else
      return norm_val * x * grad_output / beta_val;
  }
  SmoothL1BackwardFunctor(scalar_t norm_val, scalar_t beta_val)
      : norm_val(norm_val), beta_val(beta_val) {}

 private:
  scalar_t norm_val;
  scalar_t beta_val;
};

void smooth_l1_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double beta) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      iter.dtype(),
      "smooth_l1_backward_xpu",
      [&iter, &norm, beta] {
        auto norm_val = norm.to<scalar_t>();
        scalar_t beta_val(beta);
        SmoothL1BackwardFunctor<scalar_t> f(norm_val, beta_val);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t>
struct HuberBackwardFunctor {
  scalar_t operator()(scalar_t input, scalar_t target, scalar_t grad_output)
      const {
    const auto x = input - target;
    if (x < -delta_val_) {
      return -norm_val_ * grad_output * delta_val_;
    } else if (x > delta_val_) {
      return norm_val_ * grad_output * delta_val_;
    } else {
      return norm_val_ * x * grad_output;
    }
  }
  HuberBackwardFunctor(scalar_t norm_val, scalar_t delta_val)
      : norm_val_(norm_val), delta_val_(delta_val) {}

 private:
  scalar_t norm_val_;
  scalar_t delta_val_;
};

void huber_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      iter.dtype(),
      "huber_backward_xpu",
      [&iter, &norm, delta] {
        auto norm_val = norm.to<scalar_t>();
        scalar_t delta_val(delta);
        gpu_kernel(iter, HuberBackwardFunctor<scalar_t>(norm_val, delta_val));
      });
}

} // namespace at::native::xpu
