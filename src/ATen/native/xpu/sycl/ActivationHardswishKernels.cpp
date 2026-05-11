/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/XPUMathCompat.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ActivationHardswishKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct HardswishFunctor {
  scalar_t operator()(scalar_t self_val) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t zero(0.0f);
    const opmath_t one_sixth(1.0f / 6.0f);
    const opmath_t three(3.0f);
    const opmath_t six(6.0f);
    opmath_t x = static_cast<opmath_t>(self_val);
    return x * std::min(std::max(x + three, zero), six) * one_sixth;
  }
};

template <typename scalar_t>
struct HardswishBackwardFunctor {
  scalar_t operator()(scalar_t grad_val_, scalar_t self_val_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t zero(0.0f);
    const opmath_t three(3.0f);
    const opmath_t neg_three(-3.0f);
    const opmath_t one_half(0.5f);
    opmath_t grad_val = static_cast<opmath_t>(grad_val_);
    opmath_t self_val = static_cast<opmath_t>(self_val_);
    if (self_val <= neg_three) {
      return zero;
    } else if (self_val < three) {
      return grad_val * ((self_val / three) + one_half);
    } else {
      return grad_val;
    }
  }
};

void hardswish_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardswish_xpu",
      [&]() { gpu_kernel(iter, HardswishFunctor<scalar_t>()); });
}

void hardswish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardswish_backward_xpu",
      [&]() { gpu_kernel(iter, HardswishBackwardFunctor<scalar_t>()); });
}

// Functor used with multi_tensor_apply for the in-place channel-last fast path.
// The Op template must match the signature expected by UnaryOpFunctor: T(T).
template <typename T>
struct HardswishMTAFunctor {
  T operator()(T t) const {
    const T zero(0.0f);
    const T one_sixth(1.0f / 6.0f);
    const T three(3.0f);
    const T six(6.0f);
    return t * std::min(std::max(t + three, zero), six) * one_sixth;
  }
};

void hardswish_inplace_multi_tensor_kernel(TensorList tensors) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors[0].scalar_type(),
      "hardswish__xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        std::vector<std::vector<at::Tensor>> tensor_lists;
        tensor_lists.emplace_back(tensors.vec());
        multi_tensor_apply<1>(
            tensor_lists,
            UnaryOpFunctor<
                scalar_t,
                /* depth */ 1,
                /* r_args_depth */ 1,
                /* res_arg_index */ 0>(),
            HardswishMTAFunctor<opmath_t>());
      });
}

} // namespace xpu
} // namespace native
} // namespace at
