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
 * Conv3d fprop/dgrad/wgrad: im2col + Xe GEMM. Supports BF16 and FP16.
 */

#include "conv_gemm.h"

// ---------------------------------------------------------------------------
// im2col_3d
// ---------------------------------------------------------------------------

static at::Tensor im2col_3d(
    const at::Tensor& input, int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilD, int dilH, int dilW) {
  const int N = input.size(0), C = input.size(1);
  at::Tensor padded = input;
  if (padD > 0 || padH > 0 || padW > 0)
    padded = at::constant_pad_nd(input, {padW, padW, padH, padH, padD, padD}, 0.0);
  auto unfolded = padded
    .unfold(2, kD * dilD - (dilD - 1), strideD)
    .unfold(3, kH * dilH - (dilH - 1), strideH)
    .unfold(4, kW * dilW - (dilW - 1), strideW);
  if (dilD > 1 || dilH > 1 || dilW > 1)
    unfolded = unfolded.slice(5, 0, kD * dilD, dilD)
                       .slice(6, 0, kH * dilH, dilH)
                       .slice(7, 0, kW * dilW, dilW);
  const int D_out = unfolded.size(2), H_out = unfolded.size(3), W_out = unfolded.size(4);
  return unfolded.permute({0, 2, 3, 4, 1, 5, 6, 7}).contiguous()
      .reshape({N * D_out * H_out * W_out, C * kD * kH * kW});
}

// ---------------------------------------------------------------------------
// col2im_3d
// ---------------------------------------------------------------------------

static at::Tensor col2im_3d(
    const at::Tensor& columns,
    int N, int C, int D, int H, int W,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilD, int dilH, int dilW) {
  const int D_out = (D + 2 * padD - dilD * (kD - 1) - 1) / strideD + 1;
  const int H_out = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
  const int W_out = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

  // columns: [N*D_out*H_out*W_out, C*kD*kH*kW]
  // We need to use vol2col inverse. ATen doesn't have a direct 3D col2im,
  // so we implement via manual scatter.
  // Reshape to [N, D_out, H_out, W_out, C, kD, kH, kW]
  auto cols = columns.reshape({N, D_out, H_out, W_out, C, kD, kH, kW});

  // Allocate output with padding
  int D_pad = D + 2 * padD;
  int H_pad = H + 2 * padH;
  int W_pad = W + 2 * padW;
  at::Tensor output_padded = at::zeros({N, C, D_pad, H_pad, W_pad}, columns.options());

  // Scatter-add: for each output position and kernel position, accumulate
  for (int kd = 0; kd < kD; ++kd) {
    for (int kh = 0; kh < kH; ++kh) {
      for (int kw = 0; kw < kW; ++kw) {
        // cols[:, :, :, :, :, kd, kh, kw] shape [N, D_out, H_out, W_out, C]
        auto patch = cols.select(7, kw).select(6, kh).select(5, kd);
        // patch: [N, D_out, H_out, W_out, C] → [N, C, D_out, H_out, W_out]
        patch = patch.permute({0, 4, 1, 2, 3}).contiguous();

        // Build destination slice indices
        int d_start = kd * dilD, h_start = kh * dilH, w_start = kw * dilW;
        // Use a strided view to scatter
        auto dst = output_padded.slice(2, d_start, d_start + D_out * strideD, strideD)
                                .slice(3, h_start, h_start + H_out * strideH, strideH)
                                .slice(4, w_start, w_start + W_out * strideW, strideW);
        dst.add_(patch);
      }
    }
  }

  // Remove padding
  if (padD > 0 || padH > 0 || padW > 0) {
    return output_padded.slice(2, padD, padD + D)
                        .slice(3, padH, padH + H)
                        .slice(4, padW, padW + W)
                        .contiguous();
  }
  return output_padded;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

namespace sycltla {

at::Tensor conv3d_fprop_impl(
    sycl::queue& queue,
    const at::Tensor& input, const at::Tensor& weight,
    const at::Tensor& bias, bool has_bias,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilD, int dilH, int dilW,
    int64_t groups) {

  const int N = input.size(0), C = input.size(1);
  const int D = input.size(2), H = input.size(3), W = input.size(4);
  const int K = weight.size(0), C_per_group = weight.size(1);
  const int kD = weight.size(2), kH = weight.size(3), kW = weight.size(4);
  const int K_per_group = K / groups;
  const int D_out = (D + 2 * padD - dilD * (kD - 1) - 1) / strideD + 1;
  const int H_out = (H + 2 * padH - dilH * (kH - 1) - 1) / strideH + 1;
  const int W_out = (W + 2 * padW - dilW * (kW - 1) - 1) / strideW + 1;

  at::Tensor output;
  if (groups == 1) {
    at::Tensor columns = im2col_3d(input, kD, kH, kW,
        strideD, strideH, strideW, padD, padH, padW, dilD, dilH, dilW);
    at::Tensor weight_2d = weight.reshape({K, C * kD * kH * kW}).t().contiguous();
    at::Tensor output_2d = run_conv_gemm(queue, columns, weight_2d);
    output = output_2d.reshape({N, D_out, H_out, W_out, K}).permute({0, 4, 1, 2, 3}).contiguous();
  } else {
    auto input_groups = input.chunk(groups, 1);
    auto weight_groups = weight.chunk(groups, 0);
    std::vector<at::Tensor> output_groups;
    output_groups.reserve(groups);
    for (int64_t g = 0; g < groups; ++g) {
      at::Tensor cols_g = im2col_3d(input_groups[g], kD, kH, kW,
          strideD, strideH, strideW, padD, padH, padW, dilD, dilH, dilW);
      at::Tensor w2d_g = weight_groups[g].reshape({K_per_group, C_per_group * kD * kH * kW}).t().contiguous();
      at::Tensor out_g = run_conv_gemm(queue, cols_g, w2d_g);
      out_g = out_g.reshape({N, D_out, H_out, W_out, K_per_group}).permute({0, 4, 1, 2, 3});
      output_groups.push_back(out_g);
    }
    output = at::cat(output_groups, 1).contiguous();
  }

  if (has_bias && bias.defined())
    output = output + bias.reshape({1, K, 1, 1, 1});
  return output;
}

// ---------------------------------------------------------------------------
// Conv3d dgrad
// ---------------------------------------------------------------------------

at::Tensor conv3d_dgrad_impl(
    sycl::queue& queue,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    int D, int H, int W,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilD, int dilH, int dilW,
    int64_t groups) {

  const int N = grad_output.size(0), K = grad_output.size(1);
  const int D_out = grad_output.size(2), H_out = grad_output.size(3), W_out = grad_output.size(4);
  const int C_per_group = weight.size(1);
  const int C = C_per_group * groups;
  const int kD = weight.size(2), kH = weight.size(3), kW = weight.size(4);
  const int K_per_group = K / groups;

  if (groups == 1) {
    at::Tensor grad_2d = grad_output.permute({0, 2, 3, 4, 1}).contiguous()
        .reshape({N * D_out * H_out * W_out, K});
    at::Tensor weight_2d = weight.reshape({K, C * kD * kH * kW}).contiguous();
    at::Tensor columns = run_conv_gemm(queue, grad_2d, weight_2d);
    return col2im_3d(columns, N, C, D, H, W, kD, kH, kW,
        strideD, strideH, strideW, padD, padH, padW, dilD, dilH, dilW);
  } else {
    auto grad_groups = grad_output.chunk(groups, 1);
    auto weight_groups = weight.chunk(groups, 0);
    std::vector<at::Tensor> input_grad_groups;
    input_grad_groups.reserve(groups);
    for (int64_t g = 0; g < groups; ++g) {
      at::Tensor g2d = grad_groups[g].permute({0, 2, 3, 4, 1}).contiguous()
          .reshape({N * D_out * H_out * W_out, K_per_group});
      at::Tensor w2d = weight_groups[g].reshape({K_per_group, C_per_group * kD * kH * kW}).contiguous();
      at::Tensor cols = run_conv_gemm(queue, g2d, w2d);
      input_grad_groups.push_back(col2im_3d(cols, N, C_per_group, D, H, W,
          kD, kH, kW, strideD, strideH, strideW, padD, padH, padW, dilD, dilH, dilW));
    }
    return at::cat(input_grad_groups, 1).contiguous();
  }
}

// ---------------------------------------------------------------------------
// Conv3d wgrad
// ---------------------------------------------------------------------------

at::Tensor conv3d_wgrad_impl(
    sycl::queue& queue,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int dilD, int dilH, int dilW,
    int64_t groups) {

  const int N = input.size(0), C = input.size(1);
  const int K = grad_output.size(1);
  const int C_per_group = C / groups;
  const int K_per_group = K / groups;
  const int D_out = grad_output.size(2), H_out = grad_output.size(3), W_out = grad_output.size(4);

  if (groups == 1) {
    at::Tensor columns = im2col_3d(input, kD, kH, kW,
        strideD, strideH, strideW, padD, padH, padW, dilD, dilH, dilW);
    at::Tensor grad_2d = grad_output.permute({0, 2, 3, 4, 1}).contiguous()
        .reshape({N * D_out * H_out * W_out, K});
    at::Tensor cols_t = columns.t().contiguous();
    at::Tensor gw = run_conv_gemm(queue, cols_t, grad_2d);
    return gw.t().contiguous().reshape({K, C, kD, kH, kW});
  } else {
    auto input_groups = input.chunk(groups, 1);
    auto grad_groups = grad_output.chunk(groups, 1);
    std::vector<at::Tensor> wgrad_groups;
    wgrad_groups.reserve(groups);
    for (int64_t g = 0; g < groups; ++g) {
      at::Tensor cols = im2col_3d(input_groups[g], kD, kH, kW,
          strideD, strideH, strideW, padD, padH, padW, dilD, dilH, dilW);
      at::Tensor g2d = grad_groups[g].permute({0, 2, 3, 4, 1}).contiguous()
          .reshape({N * D_out * H_out * W_out, K_per_group});
      at::Tensor cols_t = cols.t().contiguous();
      at::Tensor gw = run_conv_gemm(queue, cols_t, g2d);
      wgrad_groups.push_back(gw.t().contiguous().reshape({K_per_group, C_per_group, kD, kH, kW}));
    }
    return at::cat(wgrad_groups, 0).contiguous();
  }
}

} // namespace sycltla
