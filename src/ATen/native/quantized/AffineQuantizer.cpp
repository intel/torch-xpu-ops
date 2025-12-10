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
#include <ATen/native/quantized/AffineQuantizer.h>

#include <ATen/native/quantized/sycl/AffineQuantizerKernels.h>

namespace at::native {

REGISTER_XPU_DISPATCH(
    quantize_tensor_per_tensor_affine_stub,
    &xpu::quantize_tensor_per_tensor_affine_kernel);
REGISTER_XPU_DISPATCH(
    dequantize_tensor_per_tensor_affine_stub,
    &xpu::dequantize_tensor_per_tensor_affine_kernel);
REGISTER_XPU_DISPATCH(
    quantize_tensor_per_channel_affine_stub,
    &xpu::quantize_tensor_per_channel_affine_kernel);
REGISTER_XPU_DISPATCH(
    dequantize_tensor_per_channel_affine_stub,
    &xpu::dequantize_tensor_per_channel_affine_kernel);
REGISTER_XPU_DISPATCH(
    quantize_tensor_per_channel_float_qparams_stub,
    &xpu::quantize_tensor_per_channel_float_qparams_kernel);
REGISTER_XPU_DISPATCH(
    dequantize_tensor_per_channel_float_qparams_stub,
    &xpu::dequantize_tensor_per_channel_float_qparams_kernel);

} // namespace at::native
