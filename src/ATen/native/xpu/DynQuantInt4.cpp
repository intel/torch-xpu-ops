/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ops/_dyn_quant_matmul_4bit_native.h>
#include <ATen/ops/_dyn_quant_pack_4bit_weight_native.h>
#include <ATen/native/xpu/sycl/Dequant_int4.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <comm/SYCLContext.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/matmul.h>

namespace at::native {

// Note: FP32 support was added to dequant_int4_kernel and linear_int4_kernel
// by extending AT_DISPATCH_FLOATING_TYPES_AND2 to include Half/BFloat16
// This enables mixed-precision quantization on XPU for BF16, FP16, and FP32

Tensor _dyn_quant_pack_4bit_weight_xpu(
    const Tensor& weights,
    const Tensor& scales_zeros,
    const std::optional<Tensor>& bias,
    const int64_t block_size,
    const int64_t in_features,
    const int64_t out_features) {
  // Pack on XPU: concatenate weights + scales_zeros + optional_bias
  // Format: [weights_float | scales_zeros_float | bias_float]
  // This matches CPU implementation exactly
  TORCH_CHECK(weights.dtype() == at::kByte, "weights must be uint8 packed int4");
  TORCH_CHECK(weights.device().type() == at::kXPU, "weights must be on XPU");
  TORCH_CHECK(scales_zeros.device().type() == at::kXPU, "scales_zeros must be on XPU");

  // Convert weights to float and reshape to 1D
  auto weight_reshaped = weights.reshape({-1}).to(at::kFloat);

  // Reshape scales_zeros to 1D and convert to float
  auto scales_zeros_reshaped = scales_zeros.reshape({-1}).to(at::kFloat);

  // Build list of tensors to concatenate
  std::vector<Tensor> tensors_to_cat;
  tensors_to_cat.push_back(weight_reshaped);
  tensors_to_cat.push_back(scales_zeros_reshaped);
  if (bias.has_value()) {
    tensors_to_cat.push_back(bias.value().reshape({-1}).to(at::kFloat));
  }

  // Concatenate on XPU - everything stays on device
  return at::cat(tensors_to_cat, 0);
}

Tensor _dyn_quant_matmul_4bit_xpu(
    const Tensor& inp,
    const Tensor& packed_weights,
    const int64_t block_size,
    const int64_t in_features,
    const int64_t out_features) {
  // Matmul entirely on XPU using native kernels
  TORCH_CHECK(inp.device().type() == at::kXPU, "inp must be on XPU");
  TORCH_CHECK(packed_weights.device().type() == at::kXPU, "packed_weights must be on XPU");

  int64_t M = inp.size(0);
  int64_t N = out_features;
  int64_t K = in_features;

  int64_t weights_elements = N * K / 2;  // Packed uint8 as float
  int64_t scale_elements = N * (K / block_size);

  TORCH_CHECK(
      packed_weights.numel() >= (weights_elements + scale_elements),
      "Invalid packed weight tensor size");

  // Extract components on XPU
  auto extracted_weights_float = packed_weights.narrow(0, 0, weights_elements);
  auto extracted_scales_and_bias = packed_weights.narrow(0, weights_elements, packed_weights.size(0) - weights_elements);
  auto extracted_scales = extracted_scales_and_bias.narrow(0, 0, scale_elements);

  int64_t bias_elements = packed_weights.numel() - (weights_elements + scale_elements);
  std::optional<Tensor> bias_opt = std::nullopt;
  if (bias_elements > 0) {
    bias_opt = extracted_scales_and_bias.narrow(0, scale_elements, bias_elements);
  }

  // Convert weights back from float to uint8
  Tensor weights_uint8 = extracted_weights_float.to(at::kByte);

  // Reshape weights to [N, K/2] - this is the expected input format
  Tensor weights_2d = weights_uint8.reshape({N, K / 2});

  // Prepare scale and zero_point tensor for kernel without host access.
  // The XPU kernel expects [num_blocks, N * 2] with interleaved scale/zero_point.
  int64_t num_blocks = K / block_size;
  Tensor scales_reshaped = extracted_scales.reshape({N, num_blocks}).transpose(0, 1).contiguous();
  Tensor scale_and_zeros = at::stack(
      {scales_reshaped, at::zeros_like(scales_reshaped)},
      /*dim=*/2)
      .reshape({num_blocks, N * 2})
      .contiguous();

  // Prepare output tensor on XPU
  Tensor weight_dequant = at::empty({K, N}, inp.options().dtype(inp.dtype()));

  // Call dequant_int4_kernel on XPU
  at::native::xpu::dequant_int4_kernel(
      weights_2d, weight_dequant, static_cast<int>(block_size), scale_and_zeros);

  // Perform matmul on XPU: inp (M, K) x weight_dequant (K, N) = output (M, N)
  Tensor output = inp.matmul(weight_dequant);

  // Add bias if present (stays on XPU)
  if (bias_opt.has_value()) {
    output = output + bias_opt.value();
  }

  return output;
}

} // namespace at::native