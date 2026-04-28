/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*
 * Conv2d fprop: im2col + Xe GEMM. Supports BF16 and FP16.
 */

#include <conv_gemm.h>

// ---------------------------------------------------------------------------
// im2col_2d
// ---------------------------------------------------------------------------

static at::Tensor im2col_2d(
    const at::Tensor& input,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilH,
    int dilW) {
  at::Tensor padded = input;
  if (padH > 0 || padW > 0)
    padded = at::constant_pad_nd(input, {padW, padW, padH, padH}, 0.0);
  auto unfolded = padded.unfold(2, kH * dilH - (dilH - 1), strideH)
                      .unfold(3, kW * dilW - (dilW - 1), strideW);
  if (dilH > 1 || dilW > 1)
    unfolded =
        unfolded.slice(4, 0, kH * dilH, dilH).slice(5, 0, kW * dilW, dilW);
  const int N = input.size(0);
  const int C = input.size(1);
  const int H_out = unfolded.size(2);
  const int W_out = unfolded.size(3);
  return unfolded.permute({0, 2, 3, 1, 4, 5})
      .contiguous()
      .reshape({N * H_out * W_out, C * kH * kW});
}

// ---------------------------------------------------------------------------
// Public API - fprop only
// ---------------------------------------------------------------------------

namespace sycltla {

at::Tensor conv2d_fprop_impl(
    sycl::queue& queue,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    bool has_bias,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilH,
    int dilW,
    int64_t groups) {
  const int N = input.size(0), C = input.size(1), H = input.size(2),
            W = input.size(3);
  const int K = weight.size(0), C_per_group = weight.size(1);
  const int kH = weight.size(2), kW = weight.size(3);
  const int K_per_group = K / groups;
  const int H_out = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
  const int W_out = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

  at::Tensor output;

  if (groups == 1) {
    at::Tensor columns =
        im2col_2d(input, kH, kW, strideH, strideW, padH, padW, dilH, dilW);
    at::Tensor weight_2d = weight.reshape({K, C * kH * kW}).t().contiguous();
    at::Tensor output_2d = run_conv_gemm(queue, columns, weight_2d);
    output = output_2d.reshape({N, H_out, W_out, K})
                 .permute({0, 3, 1, 2})
                 .contiguous();
  } else {
    auto input_groups = input.chunk(groups, 1);
    auto weight_groups = weight.chunk(groups, 0);
    std::vector<at::Tensor> output_groups;
    output_groups.reserve(groups);
    for (int64_t g = 0; g < groups; ++g) {
      at::Tensor columns_g = im2col_2d(
          input_groups[g], kH, kW, strideH, strideW, padH, padW, dilH, dilW);
      at::Tensor weight_2d_g =
          weight_groups[g]
              .reshape({K_per_group, C_per_group * kH * kW})
              .t()
              .contiguous();
      at::Tensor out_g = run_conv_gemm(queue, columns_g, weight_2d_g);
      out_g =
          out_g.reshape({N, H_out, W_out, K_per_group}).permute({0, 3, 1, 2});
      output_groups.push_back(out_g);
    }
    output = at::cat(output_groups, 1).contiguous();
  }

  if (has_bias && bias.defined())
    output = output + bias.reshape({1, K, 1, 1});
  return output;
}

} // namespace sycltla
