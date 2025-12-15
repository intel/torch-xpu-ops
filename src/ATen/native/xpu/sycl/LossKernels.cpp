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
#include <ATen/core/Reduction.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/LossKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct BinaryCrossEntropyFunctor {
  scalar_t operator()(scalar_t input_val, scalar_t target_val) const {
    const scalar_t zero = 0;
    const scalar_t one = 1;
    const scalar_t neg_100 = -100;

    SYCL_KERNEL_ASSERT(input_val >= zero && input_val <= one);
    SYCL_KERNEL_ASSERT(target_val >= zero && target_val <= one);

    scalar_t log_input_val = std::log(input_val);
    scalar_t log_1_minus_input_val = std::log1p(-input_val);

    log_input_val = std::max(log_input_val, neg_100);
    log_1_minus_input_val = std::max(log_1_minus_input_val, neg_100);

    return ((target_val - one) * log_1_minus_input_val) -
        (target_val * log_input_val);
  }
};

Tensor& binary_cross_entropy_kernel(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& loss) {
  Tensor loss_squeezed = at::squeeze(loss);

  TensorIterator iter = TensorIteratorConfig()
                            .add_output(loss_squeezed)
                            .add_owned_input(at::squeeze(input))
                            .add_owned_input(at::squeeze(target))
                            .build();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "binary_cross_entropy_xpu",
      [&]() { gpu_kernel(iter, BinaryCrossEntropyFunctor<scalar_t>()); });
  if (weight.defined()) {
    loss.mul_(weight);
  }

  if (reduction != at::Reduction::None) {
    Tensor loss_reduced;
    if (reduction == at::Reduction::Mean) {
      loss_reduced = loss.mean();
    } else if (reduction == at::Reduction::Sum) {
      loss_reduced = loss.sum();
    }
    loss.resize_as_(loss_reduced).copy_(loss_reduced);
  }

  return loss;
}

template <typename scalar_t>
struct BinaryCrossEntropyBackwardFunctor {
  scalar_t operator()(
      scalar_t grad_val,
      scalar_t input_val,
      scalar_t target_val) const {
    constexpr float EPSILON = 1e-12;
    const scalar_t one = 1;
    const scalar_t epsilon = EPSILON;

    scalar_t grad_input_denominator =
        std::max((one - input_val) * input_val, epsilon);

    return grad_val * (input_val - target_val) / grad_input_denominator;
  }
};

Tensor& binary_cross_entropy_backward_kernel(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& grad_input) {
  Tensor grad_expand = grad.expand_as(input);
  at::TensorIterator iter = TensorIteratorConfig()
                                .add_output(grad_input)
                                .add_input(grad_expand)
                                .add_input(input)
                                .add_input(target)
                                .build();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "binary_cross_entropy_backward_xpu",
      [&]() {
        gpu_kernel(iter, BinaryCrossEntropyBackwardFunctor<scalar_t>());
      });

  if (weight.defined()) {
    grad_input.mul_(weight);
  }
  if (reduction == at::Reduction::Mean) {
    grad_input.div_(input.numel());
  }
  return grad_input;
}

} // namespace at::native::xpu
