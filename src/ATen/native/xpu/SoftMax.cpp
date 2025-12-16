/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/SoftMaxKernels.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/_log_softmax_backward_data_native.h>
#include <ATen/ops/_log_softmax_native.h>
#include <ATen/ops/_softmax_backward_data_native.h>
#include <ATen/ops/_softmax_native.h>
namespace at::native {

TORCH_IMPL_FUNC(softmax_xpu_out)
(const Tensor& input,
 const int64_t dim,
 const bool half_to_float,
 const Tensor& output) {
  xpu::_softmax_kernel(input, dim, half_to_float, output);
}

TORCH_IMPL_FUNC(softmax_backward_xpu_out)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype,
 const Tensor& grad_input) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "xpu::_softmax_backward_data_out_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      output,
      "xpu::_softmax_backward_data_out_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, output, "xpu::_softmax_backward_data_out_out", "output");
  bool half_to_float = grad.scalar_type() != input_dtype;
  if (half_to_float) {
    TORCH_CHECK(
        (grad.scalar_type() == ScalarType::Float &&
         input_dtype == ScalarType::Half),
        "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
  }

  native::xpu::_softmax_backward_kernel(
      grad, output, dim, half_to_float, grad_input);
}

TORCH_IMPL_FUNC(log_softmax_backward_xpu_out)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype,
 const Tensor& grad_input) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "xpu::_log_softmax_backward_data_out_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      output,
      "xpu::_log_softmax_backward_data_out_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device,
      output,
      "xpu::_log_softmax_backward_data_out_out",
      "output");
  bool half_to_float = grad.scalar_type() != input_dtype;
  if (half_to_float) {
    TORCH_CHECK(
        (grad.scalar_type() == ScalarType::Float &&
         input_dtype == ScalarType::Half),
        "expected input and grad types to match, or input to be at::Half and grad to be at::Float");
  }
  native::xpu::_log_softmax_backward_kernel(
      grad, output, dim, half_to_float, grad_input);
}

TORCH_IMPL_FUNC(log_softmax_xpu_out)
(const Tensor& input,
 const int64_t dim,
 const bool half_to_float,
 const Tensor& output) {
  xpu::_log_softmax_kernel(input, dim, half_to_float, output);
}

Tensor _safe_softmax_xpu(
    const Tensor& self,
    int64_t dim,
    std::optional<ScalarType> dtype) {
  // TODO: uncomment after XPU softmax support half_to_float=true
  // if (self.scalar_type() == ScalarType::Half && dtype == ScalarType::Float)
  //   return xpu::_safe_softmax_kernel(self, dim_, true);
  Tensor converted = dtype.has_value() ? self.toType(dtype.value()) : self;
  return xpu::_safe_softmax_kernel(converted, dim, false);
}

Tensor masked_softmax_xpu(
    const Tensor& input_,
    const Tensor& mask_,
    const std::optional<int64_t> dim_,
    const std::optional<int64_t> mask_type_) {
  return xpu::masked_softmax_kernel(input_, mask_, dim_, mask_type_);
}

Tensor masked_softmax_backward_xpu(
    const Tensor& grad_,
    const Tensor& output_,
    const Tensor& mask_,
    const std::optional<int64_t> dim_) {
  return xpu::masked_softmax_backward_kernel(grad_, output_, mask_, dim_);
}

} // namespace at::native
