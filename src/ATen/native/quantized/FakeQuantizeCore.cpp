/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/native/DispatchStub.h>
#include <ATen/native/quantized/FakeQuantAffine.h>

#include <ATen/native/quantized/sycl/FakeQuantizeCoreKernels.h>

namespace at::native {

REGISTER_XPU_DISPATCH(
    fake_quant_tensor_cachemask_stub,
    &xpu::fake_quantize_tensor_cachemask_kernel)
REGISTER_XPU_DISPATCH(
    fake_quant_tensor_cachemask_tensor_qparams_stub,
    &xpu::fake_quantize_tensor_cachemask_tensor_qparams_kernel)
REGISTER_XPU_DISPATCH(
    fake_quant_grad_learnable_tensor_stub,
    &xpu::_fake_quantize_grad_learnable_tensor_kernel)
REGISTER_XPU_DISPATCH(
    fake_quant_per_channel_cachemask_stub,
    &xpu::fake_quant_per_channel_cachemask_kernel)
REGISTER_XPU_DISPATCH(
    fake_quant_grad_learnable_channel_stub,
    &xpu::_fake_quantize_grad_learnable_channel_kernel)

} // namespace at::native
