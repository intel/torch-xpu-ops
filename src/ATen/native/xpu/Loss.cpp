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
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/PointwiseOps.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/BinaryMiscOpsKernels.h>
#include <ATen/native/xpu/sycl/LossKernels.h>
#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>
#include <comm/RegisterUtils.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(mse_stub, &xpu::mse_kernel);
REGISTER_XPU_DISPATCH(mse_backward_stub, &xpu::mse_backward_kernel);
REGISTER_XPU_DISPATCH(huber_stub, &xpu::huber_kernel);
REGISTER_XPU_DISPATCH(huber_backward_stub, &xpu::huber_backward_kernel);
REGISTER_XPU_DISPATCH(smooth_l1_stub, &xpu::smooth_l1_kernel);
REGISTER_XPU_DISPATCH(smooth_l1_backward_stub, &xpu::smooth_l1_backward_kernel);

Tensor binary_cross_entropy_xpu(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  Tensor loss = at::empty_like(self);
  return native::xpu::binary_cross_entropy_kernel(
      self, target, weight, reduction, loss);
}

Tensor& binary_cross_entropy_out_xpu(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    Tensor& loss) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  return native::xpu::binary_cross_entropy_kernel(
      self, target, weight, reduction, loss);
}

Tensor binary_cross_entropy_backward_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  Tensor grad_input = at::empty_like(self);
  return native::xpu::binary_cross_entropy_backward_kernel(
      grad_output, self, target, weight, reduction, grad_input);
}

Tensor& binary_cross_entropy_backward_out_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    Tensor& grad_input) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  return native::xpu::binary_cross_entropy_backward_kernel(
      grad_output, self, target, weight, reduction, grad_input);
}

} // namespace native
} // namespace at
