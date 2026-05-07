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
 * Conv2d backward: dgrad (input gradient) + wgrad (weight gradient).
 * Uses im2col/col2im + Xe GEMM. Supports BF16 and FP16.
 */

#include <conv_gemm.h>

// ---------------------------------------------------------------------------
// im2col_2d (shared with fprop, duplicated here to keep files independent)
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
// col2im_2d: reshape columns back to spatial input
// ---------------------------------------------------------------------------

static at::Tensor col2im_2d(
    const at::Tensor& columns,
    int N,
    int C,
    int H,
    int W,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilH,
    int dilW) {
  const int H_out = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
  const int W_out = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;
  auto cols =
      columns.reshape({N, H_out * W_out, C * kH * kW}).permute({0, 2, 1});
  return at::col2im(
      cols, {H, W}, {kH, kW}, {dilH, dilW}, {padH, padW}, {strideH, strideW});
}

// ---------------------------------------------------------------------------
// Conv2d dgrad: grad_input = col2im(grad_output_2d × weight)
// ---------------------------------------------------------------------------

namespace sycltla {

at::Tensor conv2d_dgrad_impl(
    sycl::queue& queue,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    int H,
    int W,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilH,
    int dilW,
    int64_t groups) {
  const int N = grad_output.size(0), K = grad_output.size(1);
  const int H_out = grad_output.size(2), W_out = grad_output.size(3);
  const int C_per_group = weight.size(1);
  const int C = C_per_group * groups;
  const int kH = weight.size(2), kW = weight.size(3);
  const int K_per_group = K / groups;

  if (groups == 1) {
    at::Tensor grad_2d = grad_output.permute({0, 2, 3, 1})
                             .contiguous()
                             .reshape({N * H_out * W_out, K});
    at::Tensor weight_2d = weight.reshape({K, C * kH * kW}).contiguous();
    at::Tensor columns = run_conv_gemm(queue, grad_2d, weight_2d);
    return col2im_2d(
        columns, N, C, H, W, kH, kW, strideH, strideW, padH, padW, dilH, dilW);
  } else {
    auto grad_groups = grad_output.chunk(groups, 1);
    auto weight_groups = weight.chunk(groups, 0);
    std::vector<at::Tensor> input_grad_groups;
    input_grad_groups.reserve(groups);
    for (int64_t g = 0; g < groups; ++g) {
      at::Tensor grad_2d = grad_groups[g]
                               .permute({0, 2, 3, 1})
                               .contiguous()
                               .reshape({N * H_out * W_out, K_per_group});
      at::Tensor w2d = weight_groups[g]
                           .reshape({K_per_group, C_per_group * kH * kW})
                           .contiguous();
      at::Tensor cols = run_conv_gemm(queue, grad_2d, w2d);
      input_grad_groups.push_back(col2im_2d(
          cols,
          N,
          C_per_group,
          H,
          W,
          kH,
          kW,
          strideH,
          strideW,
          padH,
          padW,
          dilH,
          dilW));
    }
    return at::cat(input_grad_groups, 1).contiguous();
  }
}

// ---------------------------------------------------------------------------
// Conv2d wgrad: grad_weight = im2col(input)^T × grad_output_2d
// ---------------------------------------------------------------------------

at::Tensor conv2d_wgrad_impl(
    sycl::queue& queue,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int dilH,
    int dilW,
    int64_t groups) {
  const int N = input.size(0), C = input.size(1);
  const int K = grad_output.size(1);
  const int C_per_group = C / groups;
  const int K_per_group = K / groups;
  const int H_out = grad_output.size(2), W_out = grad_output.size(3);

  if (groups == 1) {
    at::Tensor columns =
        im2col_2d(input, kH, kW, strideH, strideW, padH, padW, dilH, dilW);
    at::Tensor grad_2d = grad_output.permute({0, 2, 3, 1})
                             .contiguous()
                             .reshape({N * H_out * W_out, K});
    at::Tensor columns_t = columns.t().contiguous();
    at::Tensor grad_weight_2d = run_conv_gemm(queue, columns_t, grad_2d);
    return grad_weight_2d.t().contiguous().reshape({K, C, kH, kW});
  } else {
    auto input_groups = input.chunk(groups, 1);
    auto grad_groups = grad_output.chunk(groups, 1);
    std::vector<at::Tensor> wgrad_groups;
    wgrad_groups.reserve(groups);
    for (int64_t g = 0; g < groups; ++g) {
      at::Tensor cols = im2col_2d(
          input_groups[g], kH, kW, strideH, strideW, padH, padW, dilH, dilW);
      at::Tensor grad_2d = grad_groups[g]
                               .permute({0, 2, 3, 1})
                               .contiguous()
                               .reshape({N * H_out * W_out, K_per_group});
      at::Tensor cols_t = cols.t().contiguous();
      at::Tensor gw = run_conv_gemm(queue, cols_t, grad_2d);
      wgrad_groups.push_back(
          gw.t().contiguous().reshape({K_per_group, C_per_group, kH, kW}));
    }
    return at::cat(wgrad_groups, 0).contiguous();
  }
}

} // namespace sycltla
