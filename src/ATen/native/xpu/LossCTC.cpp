/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/LossCTCKernels.h>
#include <comm/RegisterUtils.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> ctc_loss_xpu(
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    int64_t blank,
    bool zero_infinity) {
  return native::xpu::ctc_loss_kernel(
      log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}

Tensor ctc_loss_backward_xpu(
    const Tensor& grad,
    const Tensor& log_probs,
    const Tensor& targets,
    IntArrayRef input_lengths,
    IntArrayRef target_lengths,
    const Tensor& neg_log_likelihood,
    const Tensor& log_alpha,
    int64_t blank,
    bool zero_infinity) {
  return native::xpu::ctc_loss_backward_kernel(
      grad,
      log_probs,
      targets,
      input_lengths,
      target_lengths,
      neg_log_likelihood,
      log_alpha,
      blank,
      zero_infinity);
}

} // namespace native
} // namespace at
