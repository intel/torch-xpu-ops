/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/nested/NestedTensorMath.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/SymIntArrayRef.h>
#include <comm/xpu_aten.h>

// #ifndef AT_PER_OPERATOR_HEADERS
// #include <ATen/Functions.h>
// #include <ATen/NativeFunctions.h>
// #else
// #include <ATen/ops/empty.h>
// #endif

#include <ATen/native/xpu/sycl/LayerNormKernels.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/zeros_like_native.h>

namespace at {
namespace native {
::std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_xpu(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const ::std::optional<at::Tensor>& weight_opt,
    const ::std::optional<at::Tensor>& bias_opt,
    double epsilon) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::native_layer_norm", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight_opt, "xpu::native_layer_norm", "weight_opt");
  c10::impl::check_and_update_common_device(
      common_device, bias_opt, "xpu::native_layer_norm", "bias_opt");

  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = at::native::_check_layer_norm_inputs(
      input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor Y = at::native::empty_like(
      *X,
      std::nullopt /* dtype */,
      std::nullopt /* layout */,
      std::nullopt /* device */,
      std::nullopt /* pin_memory */,
      LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  auto acc_type = at::toAccumulateType(input.scalar_type(), kXPU);

  Tensor mean = at::empty({M}, X->options().dtype(acc_type));
  Tensor rstd = at::empty({M}, X->options().dtype(acc_type));

  native::xpu::layer_norm_kernel(
      *X, *gamma, *beta, M, N, epsilon, &Y, &mean, &rstd);

  const auto input_shape = input.sizes();
  const size_t axis = input.dim() - normalized_shape.size();

  std::vector<int64_t> stat_shape;
  for (const auto idx : c10::irange(axis)) {
    stat_shape.push_back(input_shape[idx]);
  }
  for ([[maybe_unused]] const auto _ : c10::irange(axis, input.dim())) {
    stat_shape.push_back(1);
  }

  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);

  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_xpu(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const ::std::optional<at::Tensor>& weight_opt,
    const ::std::optional<at::Tensor>& bias_opt,
    ::std::array<bool, 3> grad_input_mask) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, grad_output, "xpu::native_layer_norm_backward", "goutput");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::native_layer_norm_backward", "input");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::native_layer_norm_backward", "mean");
  c10::impl::check_and_update_common_device(
      common_device, rstd, "xpu::native_layer_norm_backward", "rstd");
  c10::impl::check_and_update_common_device(
      common_device, weight_opt, "xpu::native_layer_norm_backward", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias_opt, "xpu::native_layer_norm_backward", "bias");

  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = at::native::_check_layer_norm_inputs(
      input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  if (grad_input_mask[0]) {
    grad_input = at::native::empty_like(
        *X,
        std::nullopt /* dtype */,
        std::nullopt /* layout */,
        std::nullopt /* device */,
        std::nullopt /* pin_memory */,
        LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  if (grad_input_mask[1]) {
    grad_weight = M > 0 ? at::native::empty_like(
                              *gamma,
                              std::nullopt /* dtype */,
                              std::nullopt /* layout */,
                              std::nullopt /* device */,
                              std::nullopt /* pin_memory */,
                              LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                        : at::native::zeros_like(
                              *gamma,
                              std::nullopt /* dtype */,
                              std::nullopt /* layout */,
                              std::nullopt /* device */,
                              std::nullopt /* pin_memory */,
                              LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  if (grad_input_mask[2]) {
    grad_bias = M > 0 ? at::native::empty_like(
                            *beta,
                            std::nullopt /* dtype */,
                            std::nullopt /* layout */,
                            std::nullopt /* device */,
                            std::nullopt /* pin_memory */,
                            LEGACY_CONTIGUOUS_MEMORY_FORMAT)
                      : at::native::zeros_like(
                            *beta,
                            std::nullopt /* dtype */,
                            std::nullopt /* layout */,
                            std::nullopt /* device */,
                            std::nullopt /* pin_memory */,
                            LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  return native::xpu::layer_norm_backward_kernel(
      grad_output.contiguous(),
      *X,
      mean,
      rstd,
      *gamma,
      M,
      N,
      grad_input,
      grad_weight,
      grad_bias,
      grad_input_mask);
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_nested_xpu(
    const Tensor& grad,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /*{ optional */,
    std::array<bool, 3> grad_input_mask) {
  TORCH_CHECK_VALUE(weight_opt.has_value() && bias_opt.has_value(), "NestedTensor layer_norm requires weight and bias");
  // For NestedTensors weight and bias are non nested.
  auto* nt_impl_grad = get_nested_tensor_impl(grad);
  auto* nt_impl_input = get_nested_tensor_impl(input);
  const auto& weight = weight_opt.value();
  const auto& bias = bias_opt.value();
  const auto& sizes = nt_impl_input->get_nested_sizes();
  auto M_N = _check_nested_layer_norm_inputs(
      *nt_impl_input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;

  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  Tensor dInput;
  Tensor dgamma;
  Tensor dbeta;
  auto input_buffer = nt_impl_input->get_buffer();
  auto grad_buffer = nt_impl_grad->get_buffer();
  if (grad_input_mask[0]) {
    dInput = at::native::empty_like(
        input_buffer,
        std::nullopt /* dtype */,
        std::nullopt /* layout */,
        std::nullopt /* device */,
        std::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  } else {
    dInput = at::native::zeros_like(
        input_buffer,
        std::nullopt /* dtype */,
        std::nullopt /* layout */,
        std::nullopt /* device */,
        std::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[1]) {
    dgamma = M > 0 ? at::native::empty_like(
                         *gamma,
                         std::nullopt /* dtype */,
                         std::nullopt /* layout */,
                         std::nullopt /* device */,
                         std::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous)
                   : at::native::zeros_like(
                         *gamma,
                         std::nullopt /* dtype */,
                         std::nullopt /* layout */,
                         std::nullopt /* device */,
                         std::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    dbeta = M > 0 ? at::native::empty_like(
                        *beta,
                        std::nullopt /* dtype */,
                        std::nullopt /* layout */,
                        std::nullopt /* device */,
                        std::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous)
                  : at::native::zeros_like(
                        *beta,
                        std::nullopt /* dtype */,
                        std::nullopt /* layout */,
                        std::nullopt /* device */,
                        std::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous);
  }
  native::xpu::layer_norm_backward_kernel(
      grad_buffer,
      input_buffer,
      mean,
      rstd,
      *gamma,
      M,
      N,
      dInput,
      dgamma,
      dbeta,
      grad_input_mask);

    return std::make_tuple(wrap_buffer(dInput, sizes), std::move(dgamma), std::move(dbeta));
}


REGISTER_XPU_DISPATCH(LayerNormKernel, &xpu::layer_norm_kernel);
} // namespace native

} // namespace at
