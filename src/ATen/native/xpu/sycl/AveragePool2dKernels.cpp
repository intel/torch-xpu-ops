/*
 * Copyright 2020-2026 Intel Corporation
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

#include <algorithm>

#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Pool.h>

#include <ATen/native/xpu/sycl/AveragePool2dKernels.h>
#include <ATen/native/xpu/sycl/IntegerDivider.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/xpu/XPUContext.h>
#ifdef USE_SYCLTLA
#define CUTLASS_SYCL_BUILTIN_ENABLE
// Keep dependency-header diagnostics separate from kernel diagnostics.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#endif
#include <cute/tensor.hpp>

#include <cute/atom/copy_traits_xe_2d.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

namespace at::native {
namespace xpu {

namespace avg_pool_backward {
#ifdef USE_SYCLTLA
bool can_use_tla_block_io() {
  const auto arch =
      c10::xpu::getCurrentXPUStream()
          .queue()
          .get_device()
          .get_info<
              sycl::ext::oneapi::experimental::info::device::architecture>();
  using Architecture = sycl::ext::oneapi::experimental::architecture;
  return arch == Architecture::intel_gpu_bmg_g21 ||
      arch == Architecture::intel_gpu_bmg_g31;
}
#endif

template <typename scalar_t, int V>
void store(scalar_t* p, memory::aligned_vector<scalar_t, V> v) {
  *reinterpret_cast<memory::aligned_vector<scalar_t, V>*>(p) = v;
}
template <typename scalar_t, int K, int S, int P, bool Include, int V>
struct Tile {
  const scalar_t* src;
  scalar_t* dst;
  int C, H, W, OH, OW, CS, tx, ty, count;
  at::detail::IntDivider<uint32_t> dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const {
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int q = dc.div(id), c = (id - q * dc.divisor) * V;
    int r = dx.div(q), x = (q - r * dx.divisor) * 2;
    int n = dy.div(r), y = (r - n * dy.divisor) * 2;
    float acc[4][V]{};
    constexpr int span = K == 3 ? (S == 1 ? 4 : 2) : 1;
    int sy0 = (S == 1) ? y - P : y / S;
    int sx0 = (S == 1) ? x - P : x / S;
#pragma unroll
    for (int ry = 0; ry < span; ++ry) {
      int sy = sy0 + ry;
#pragma unroll
      for (int rx = 0; rx < span; ++rx) {
        int sx = sx0 + rx;
        if (sy >= 0 && sy < OH && sx >= 0 && sx < OW) {
          float divisor;
          if constexpr (Include) {
            if constexpr (K == 3)
              divisor = 9.f;
            else
              divisor = float(
                  sycl::min(K, H + P - (sy * S - P)) *
                  sycl::min(K, W + P - (sx * S - P)));
          } else
            divisor = float(
                (sycl::min(sy * S - P + K, H) - sycl::max(sy * S - P, 0)) *
                (sycl::min(sx * S - P + K, W) - sycl::max(sx * S - P, 0)));
          memory::aligned_vector<scalar_t, V> v;
          v = *reinterpret_cast<const decltype(v)*>(
              src + ((n * OH + sy) * OW + sx) * CS + c);
          float terms[V];
#pragma unroll
          for (int j = 0; j < V; ++j) {
            if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
              const float inv = divisor == 4.f ? .25f
                  : divisor == 6.f             ? 1.f / 6.f
                  : divisor == 9.f             ? 1.f / 9.f
                                               : 1.f / divisor;
              terms[j] = float(scalar_t(float(v[j]) * inv));
            } else {
              terms[j] = float(scalar_t(float(v[j]) / divisor));
            }
          }
#pragma unroll
          for (int yy = 0; yy < 2; ++yy) {
#pragma unroll
            for (int xx = 0; xx < 2; ++xx) {
              if (y + yy >= sy * S - P && y + yy < sy * S - P + K &&
                  x + xx >= sx * S - P && x + xx < sx * S - P + K) {
#pragma unroll
                for (int j = 0; j < V; ++j)
                  acc[yy * 2 + xx][j] += terms[j];
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int yy = 0; yy < 2; ++yy) {
#pragma unroll
      for (int xx = 0; xx < 2; ++xx) {
        if (y + yy < H && x + xx < W) {
          memory::aligned_vector<scalar_t, V> v;
#pragma unroll
          for (int j = 0; j < V; ++j)
            v[j] = scalar_t(acc[yy * 2 + xx][j]);
          store<scalar_t, V>(dst + ((n * H + y + yy) * W + x + xx) * C + c, v);
        }
      }
    }
  }
};

template <typename scalar_t, int K, int S, int P, bool Include, int V>
struct Linear {
  const scalar_t* src;
  scalar_t* dst;
  int C, H, W, OH, OW, CS, count;
  at::detail::IntDivider<uint32_t> dc, dw, dh;
  void operator()(sycl::nd_item<1> it) const {
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int pix = dc.div(id), c = (id - pix * dc.divisor) * V;
    int row = dw.div(pix), x = pix - row * W;
    int n = dh.div(row), y = row - n * H;
    float acc[V]{};
    int sy0 = sycl::max(0, (y + P - K + S) / S),
        sy1 = sycl::min(OH, (y + P) / S + 1);
    int sx0 = sycl::max(0, (x + P - K + S) / S),
        sx1 = sycl::min(OW, (x + P) / S + 1);
    for (int sy = sy0; sy < sy1; ++sy)
      for (int sx = sx0; sx < sx1; ++sx) {
        float div;
        if constexpr (Include && K == 3)
          div = 9.f;
        else if constexpr (K == 2)
          div = 4.f;
        else if constexpr (K == 1)
          div = 1.f;
        else
          div = float(
              (sycl::min(sy * S - P + K, H) - sycl::max(sy * S - P, 0)) *
              (sycl::min(sx * S - P + K, W) - sycl::max(sx * S - P, 0)));
        memory::aligned_vector<scalar_t, V> v;
        v = *reinterpret_cast<const decltype(v)*>(
            src + ((n * OH + sy) * OW + sx) * CS + c);
#pragma unroll
        for (int j = 0; j < V; ++j)
          acc[j] += float(scalar_t(
              float(v[j]) *
              (div == 4.f       ? .25f
                   : div == 6.f ? 1.f / 6.f
                   : div == 9.f ? 1.f / 9.f
                                : 1.f / div)));
      }
    memory::aligned_vector<scalar_t, V> out;
#pragma unroll
    for (int j = 0; j < V; ++j)
      out[j] = scalar_t(acc[j]);
    store<scalar_t, V>(dst + id * V, out);
  }
};
template <typename scalar_t, int K, int S, int P, bool Include, int V>
void launch(at::Tensor go, at::Tensor out) {
  if (out.size(1) % 32 != 0) {
    int C = out.size(1), H = out.size(2), W = out.size(3),
        count = out.numel() / V;
    c10::xpu::getCurrentXPUStream().queue().parallel_for(
        sycl::nd_range<1>((count + 255) / 256 * 256, 256),
        Linear<scalar_t, K, S, P, Include, V>{
            reinterpret_cast<const scalar_t*>(go.data_ptr()),
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            C,
            H,
            W,
            int(go.size(2)),
            int(go.size(3)),
            int(go.stride(3)),
            count,
            at::detail::IntDivider<uint32_t>(C / V),
            at::detail::IntDivider<uint32_t>(W),
            at::detail::IntDivider<uint32_t>(H)});
    return;
  }
  int C = out.size(1), H = out.size(2), W = out.size(3), tx = (W + 1) / 2,
      ty = (H + 1) / 2;
  int count = out.size(0) * tx * ty * (C / V);
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(
      sycl::nd_range<1>((count + 255) / 256 * 256, 256),
      Tile<scalar_t, K, S, P, Include, V>{
          reinterpret_cast<const scalar_t*>(go.data_ptr()),
          reinterpret_cast<scalar_t*>(out.data_ptr()),
          C,
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          int(go.stride(3)),
          tx,
          ty,
          count,
          at::detail::IntDivider<uint32_t>(C / V),
          at::detail::IntDivider<uint32_t>(tx),
          at::detail::IntDivider<uint32_t>(ty)});
}
template <typename scalar_t, int K, int S, int P, bool Include>
struct Planar {
  const scalar_t* src;
  scalar_t* dst;
  int H, W, OH, OW, total;
  at::detail::IntDivider<uint32_t> dp, dw;
  void operator()(sycl::nd_item<1> it) const {
    int start = it.get_global_linear_id() * 8;
    if (start >= total)
      return;
    memory::aligned_vector<scalar_t, 8> v;
    if constexpr (K == 2) {
      int nc = dp.div(start), pos = start - nc * H * W, y = dw.div(pos),
          x = pos - y * W;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        scalar_t a = scalar_t(
            float(src[nc * OH * OW + (y / 2) * OW + x / 2 + j]) * .25f);
        v[j * 2] = a;
        v[j * 2 + 1] = a;
      }
      store<scalar_t, 8>(dst + start, v);
      return;
    }

#pragma unroll
    for (int j = 0; j < 8; j++) {
      int index = start + j, nc = dp.div(index), pos = index - nc * H * W,
          y = dw.div(pos), x = pos - y * W;
      float acc = 0;
      int sy0 = sycl::max(0, (y + P - K + S) / S),
          sy1 = sycl::min(OH, (y + P) / S + 1),
          sx0 = sycl::max(0, (x + P - K + S) / S),
          sx1 = sycl::min(OW, (x + P) / S + 1);
      for (int sy = sy0; sy < sy1; sy++)
        for (int sx = sx0; sx < sx1; sx++) {
          float div;
          if constexpr (Include)
            div = float(K * K);
          else
            div = float(
                (sycl::min(sy * S - P + K, H) - sycl::max(sy * S - P, 0)) *
                (sycl::min(sx * S - P + K, W) - sycl::max(sx * S - P, 0)));
          acc += float(scalar_t(
              float(src[nc * OH * OW + sy * OW + sx]) *
              (div == 4.f       ? .25f
                   : div == 6.f ? 1.f / 6.f
                                : 1.f / 9.f)));
        }
      v[j] = scalar_t(acc);
    }
    store<scalar_t, 8>(dst + start, v);
  }
};
template <typename scalar_t, int K, int S, int P, bool Include>
void planar_launch(at::Tensor go, at::Tensor out) {
  int H = out.size(2), W = out.size(3), n = out.numel();
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((n + 2047) / 2048 * 256, 256),
      Planar<scalar_t, K, S, P, Include>{
          reinterpret_cast<const scalar_t*>(go.data_ptr()),
          reinterpret_cast<scalar_t*>(out.data_ptr()),
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          n,
          at::detail::IntDivider<uint32_t>(H * W),
          at::detail::IntDivider<uint32_t>(W)});
}

template <typename scalar_t, int K, int S, int P, bool Include>
struct PackedLinear {
  const scalar_t* src;
  scalar_t* dst;
  int C, H, W, OH, OW, CS, total;
  at::detail::IntDivider<uint32_t> dc, dw, dh;
  void operator()(sycl::nd_item<1> it) const {
    int start = it.get_global_linear_id() * 8;
    if (start >= total)
      return;
    int pixel = dc.div(start), c = start - pixel * C, row = dw.div(pixel),
        x = pixel - row * W, n = dh.div(row), y = row - n * H;
    memory::aligned_vector<scalar_t, 8> result;
#pragma unroll
    for (int pair = 0; pair < 4; pair++) {
      int ch = c + pair * 2, xx = x, yy = y, nn = n;
      if (ch >= C) {
        ch -= C;
        if (++xx == W) {
          xx = 0;
          if (++yy == H) {
            yy = 0;
            ++nn;
          }
        }
      }
      float a = 0, b = 0;
      int sy0 = sycl::max(0, (yy + P - K + S) / S),
          sy1 = sycl::min(OH, (yy + P) / S + 1),
          sx0 = sycl::max(0, (xx + P - K + S) / S),
          sx1 = sycl::min(OW, (xx + P) / S + 1);
      for (int sy = sy0; sy < sy1; sy++)
        for (int sx = sx0; sx < sx1; sx++) {
          float inv;
          if constexpr (K == 3 && Include)
            inv = 1.f / 9.f;
          else if constexpr (K == 2)
            inv = .25f;
          else if constexpr (K == 1)
            inv = 1.f;
          else {
            int d = (sycl::min(sy * S - P + K, H) - sycl::max(sy * S - P, 0)) *
                (sycl::min(sx * S - P + K, W) - sycl::max(sx * S - P, 0));
            inv = d == 4 ? .25f : d == 6 ? 1.f / 6.f : 1.f / 9.f;
          }
          memory::aligned_vector<scalar_t, 2> v;
          v = *reinterpret_cast<const decltype(v)*>(
              src + ((nn * OH + sy) * OW + sx) * CS + ch);
          a += float(scalar_t(float(v[0]) * inv));
          b += float(scalar_t(float(v[1]) * inv));
        }
      result[pair * 2] = scalar_t(a);
      result[pair * 2 + 1] = scalar_t(b);
    }
    store<scalar_t, 8>(dst + start, result);
  }
};
template <typename scalar_t, int K, int S, int P, bool Include>
void packed_launch(at::Tensor go, at::Tensor out) {
  int n = out.numel(), C = out.size(1), H = out.size(2), W = out.size(3);
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((n + 2047) / 2048 * 256, 256),
      PackedLinear<scalar_t, K, S, P, Include>{
          reinterpret_cast<const scalar_t*>(go.data_ptr()),
          reinterpret_cast<scalar_t*>(out.data_ptr()),
          C,
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          int(go.stride(3)),
          n,
          at::detail::IntDivider<uint32_t>(C),
          at::detail::IntDivider<uint32_t>(W),
          at::detail::IntDivider<uint32_t>(H)});
}

template <typename scalar_t, int K, int S, int P, bool Include, int V>
struct WideTile {
  const scalar_t* src;
  scalar_t* dst;
  int C, H, W, OH, OW, CS, tx, ty, count;
  at::detail::IntDivider<uint32_t> dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const {
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int q = dc.div(id), c = (id - q * dc.divisor) * V;
    int r = dx.div(q), x = (q - r * dx.divisor) * 4;
    int n = dy.div(r), y = (r - n * dy.divisor) * 2;
    float acc[8][V]{};
    constexpr int span = K == 3 ? (S == 1 ? 4 : 2) : 1;
    int sy0 = (S == 1) ? y - P : y / S;
    int sx0 = (S == 1) ? x - P : x / S;
#pragma unroll
    for (int ry = 0; ry < span; ++ry) {
      int sy = sy0 + ry;
#pragma unroll
      for (int rx = 0; rx < 6; ++rx) {
        int sx = sx0 + rx;
        if (sy >= 0 && sy < OH && sx >= 0 && sx < OW) {
          float divisor;
          if constexpr (Include) {
            if constexpr (K == 3)
              divisor = 9.f;
            else
              divisor = float(
                  sycl::min(K, H + P - (sy * S - P)) *
                  sycl::min(K, W + P - (sx * S - P)));
          } else
            divisor = float(
                (sycl::min(sy * S - P + K, H) - sycl::max(sy * S - P, 0)) *
                (sycl::min(sx * S - P + K, W) - sycl::max(sx * S - P, 0)));
          memory::aligned_vector<scalar_t, V> v;
          v = *reinterpret_cast<const decltype(v)*>(
              src + ((n * OH + sy) * OW + sx) * CS + c);
          float terms[V];
#pragma unroll
          for (int j = 0; j < V; ++j) {
            if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
              const float inv = divisor == 4.f ? .25f
                  : divisor == 6.f             ? 1.f / 6.f
                  : divisor == 9.f             ? 1.f / 9.f
                                               : 1.f / divisor;
              terms[j] = float(scalar_t(float(v[j]) * inv));
            } else {
              terms[j] = float(scalar_t(float(v[j]) / divisor));
            }
          }
#pragma unroll
          for (int yy = 0; yy < 2; ++yy) {
#pragma unroll
            for (int xx = 0; xx < 4; ++xx) {
              if (y + yy >= sy * S - P && y + yy < sy * S - P + K &&
                  x + xx >= sx * S - P && x + xx < sx * S - P + K) {
#pragma unroll
                for (int j = 0; j < V; ++j)
                  acc[yy * 4 + xx][j] += terms[j];
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int yy = 0; yy < 2; ++yy) {
#pragma unroll
      for (int xx = 0; xx < 4; ++xx) {
        if (y + yy < H && x + xx < W) {
          memory::aligned_vector<scalar_t, V> v;
#pragma unroll
          for (int j = 0; j < V; ++j)
            v[j] = scalar_t(acc[yy * 4 + xx][j]);
          store<scalar_t, V>(dst + ((n * H + y + yy) * W + x + xx) * C + c, v);
        }
      }
    }
  }
};

template <typename scalar_t, int K, int S, int P, bool Include>
void wide_launch(at::Tensor go, at::Tensor out) {
  int C = out.size(1), H = out.size(2), W = out.size(3), tx = (W + 3) / 4,
      ty = (H + 1) / 2;
  int count = out.size(0) * tx * ty * (C / 4);
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 255) / 256 * 256, 256),
      WideTile<scalar_t, K, S, P, Include, 4>{
          reinterpret_cast<const scalar_t*>(go.data_ptr()),
          reinterpret_cast<scalar_t*>(out.data_ptr()),
          C,
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          int(go.stride(3)),
          tx,
          ty,
          count,
          at::detail::IntDivider<uint32_t>(C / 4),
          at::detail::IntDivider<uint32_t>(tx),
          at::detail::IntDivider<uint32_t>(ty)});
}
template <typename scalar_t, int V>
struct StrideTwo {
  const scalar_t* src;
  scalar_t* dst;
  int C, H, W, OH, OW, CS, count;
  at::detail::IntDivider<uint32_t> dc, dw, dh;
  memory::aligned_vector<float, V> get(int n, int y, int x, int c) const {
    memory::aligned_vector<scalar_t, V> h;
    h = *reinterpret_cast<const decltype(h)*>(
        src + ((n * OH + y) * OW + x) * CS + c);
    memory::aligned_vector<float, V> f;
#pragma unroll
    for (int j = 0; j < V; j++)
      f[j] = float(scalar_t(float(h[j]) * (1.f / 9.f)));
    return f;
  }
  void operator()(sycl::nd_item<1> it) const {
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int pix = dc.div(id), c = (id - pix * dc.divisor) * V, row = dw.div(pix),
        x = pix - row * W, n = dh.div(row), y = row - n * H, sy = y / 2,
        sx = x / 2;
    auto sum = get(n, sy, sx, c);
    bool right = (x % 2) && sx + 1 < OW, bottom = (y % 2) && sy + 1 < OH;
    if (right)
      sum += get(n, sy, sx + 1, c);
    if (bottom) {
      sum += get(n, sy + 1, sx, c);
      if (right)
        sum += get(n, sy + 1, sx + 1, c);
    }
    memory::aligned_vector<scalar_t, V> out;
#pragma unroll
    for (int j = 0; j < V; j++)
      out[j] = scalar_t(sum[j]);
    store<scalar_t, V>(dst + id * V, out);
  }
};
template <typename scalar_t, int V>
void stride_two_launch(at::Tensor go, at::Tensor out) {
  int C = out.size(1), H = out.size(2), W = out.size(3),
      count = out.numel() / V;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 255) / 256 * 256, 256),
      StrideTwo<scalar_t, V>{
          reinterpret_cast<const scalar_t*>(go.data_ptr()),
          reinterpret_cast<scalar_t*>(out.data_ptr()),
          C,
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          int(go.stride(3)),
          count,
          at::detail::IntDivider<uint32_t>(C / V),
          at::detail::IntDivider<uint32_t>(W),
          at::detail::IntDivider<uint32_t>(H)});
}

#ifdef USE_SYCLTLA
template <typename scalar_t, int W, int S, int Rows = 4, int Channels = 64>
struct AvgPoolMixedBlockTile {
  const scalar_t* src;
  scalar_t* dst;
  int C;
  at::detail::IntDivider<uint32_t> dc, dy;
  sycl::local_accessor<uint32_t, 1> tile;
  static constexpr int OH = W / S;
  static constexpr int InputRows = Rows / S + (S == 1 ? 2 : 1);
  static constexpr int InputSize = (InputRows * OH + 15) / 16 * 16;
  static constexpr int Plane = Rows * W / 2 + 1;
  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const int lid = it.get_local_linear_id(), g = it.get_group_linear_id();
    const int row = dc.div(g), cb = (g - row * dc.divisor) * Channels;
    const int n = dy.div(row), y0 = (row - n * dy.divisor) * Rows;
    const int sy0 = y0 / S - (S == 1 ? 1 : 0);
    using Term = std::conditional_t<S == 1, scalar_t, uint32_t>;
    Term normalized[S == 1 ? InputSize : InputSize / 2];
#pragma unroll
    for (int b = 0; b < InputSize / 16; ++b) {
      if constexpr (S == 1) {
        memory::aligned_vector<scalar_t, 16> input;
        cute::XE_2D_U16x16x16_LD_N::copy(
            src + n * OH * OH * C,
            C * 2,
            OH * OH,
            C * 2,
            cute::intel::coord_t{cb + lid / 16 * 16, sy0 * OH + b * 16},
            input.val);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const int sy = sy0 + (b * 16 + j) / OH, sx = (b * 16 + j) % OH;
          const int d = (sy == 0 || sy == OH - 1 ? 2 : 3) *
              (sx == 0 || sx == OH - 1 ? 2 : 3);
          const float inv = d == 4 ? .25f : d == 6 ? 1.f / 6.f : 1.f / 9.f;
          normalized[b * 16 + j] = sy >= 0 && sy < OH
              ? scalar_t(float(input[j]) * inv)
              : scalar_t(0);
        }
      } else {
        memory::aligned_vector<uint32_t, 8> input;
        cute::XE_2D_U16x16x16_LD_V::copy(
            src + n * OH * OH * C,
            C * 2,
            OH * OH,
            C * 2,
            cute::intel::coord_t{cb + lid / 16 * 16, sy0 * OH + b * 16},
            reinterpret_cast<scalar_t*>(input.val));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          auto pair =
              sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(input[j]);
#pragma unroll
          for (int q = 0; q < 2; ++q)
            pair[q] = sy0 * OH + b * 16 + j * 2 + q < OH * OH
                ? scalar_t(float(pair[q]) * (1.f / 9.f))
                : scalar_t(0);
          normalized[b * 8 + j] = sycl::bit_cast<uint32_t>(pair);
        }
      }
    }
    auto value = [&](int index) {
      if constexpr (S == 1)
        return float(normalized[index]);
      else
        return float(sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
            normalized[index / 2])[index % 2]);
    };
    float vertical[Rows * W];
    if constexpr (S == 1) {
#pragma unroll
      for (int y = 0; y < Rows; ++y) {
#pragma unroll
        for (int x = 0; x < W; ++x)
          vertical[y * W + x] = (value(y * W + x) + value((y + 1) * W + x)) +
              value((y + 2) * W + x);
      }
    }
#pragma unroll
    for (int y = 0; y < Rows; ++y) {
#pragma unroll
      for (int x = 0; x < W; x += 2) {
        memory::aligned_vector<scalar_t, 2> result;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          float sum = 0;
          if constexpr (S == 1) {
            sum = (x + j > 0 ? vertical[y * W + x + j - 1] : 0.f) +
                vertical[y * W + x + j] +
                (x + j + 1 < W ? vertical[y * W + x + j + 1] : 0.f);
          } else {
            const int sy = y / 2, sx = (x + j) / 2;
            sum = value(sy * OH + sx);
            if (j && sx + 1 < OH)
              sum += value(sy * OH + sx + 1);
            if (y % 2) {
              sum += value((sy + 1) * OH + sx);
              if (j && sx + 1 < OH)
                sum += value((sy + 1) * OH + sx + 1);
            }
          }
          result[j] = scalar_t(sum);
        }
        tile[lid * Plane + (y * W + x) / 2] = sycl::bit_cast<uint32_t>(result);
      }
    }
    if constexpr (S == 1)
      it.barrier(sycl::access::fence_space::local_space);
    else
      sycl::group_barrier(it.get_sub_group());
    constexpr int StoreThreads = S == 1 ? Channels : 16;
    const int lane = S == 1 ? lid : lid % 16;
    const int channel_base = S == 1 ? 0 : lid / 16 * 16;
    if constexpr (Rows == W) {
      for (int i = lane; i < StoreThreads * W * W / 8; i += StoreThreads) {
        const int pos = i * 4;
        memory::aligned_vector<uint32_t, 4> packed;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          packed[j] = tile
              [(channel_base + (pos + j) / (W * W / 2)) * Plane +
               (pos + j) % (W * W / 2)];
        *reinterpret_cast<decltype(packed)*>(
            dst + (n * C + cb + channel_base) * W * W + pos * 2) = packed;
      }
      return;
    }
    constexpr int Pack = Rows * W % 8 == 0 ? 4 : 2;
    constexpr int Vectors = Rows * W / (2 * Pack);
    for (int i = lane; i < StoreThreads * Vectors; i += StoreThreads) {
      const int ch = channel_base + i / Vectors, pos = i % Vectors * (2 * Pack);
      if (cb + ch < C && y0 * W + pos < W * W) {
        memory::aligned_vector<uint32_t, Pack> packed;
#pragma unroll
        for (int j = 0; j < Pack; ++j)
          packed[j] = tile[ch * Plane + pos / 2 + j];
        *reinterpret_cast<decltype(packed)*>(
            dst + (n * C + cb + ch) * W * W + y0 * W + pos) = packed;
      }
    }
  }
};
template <typename scalar_t, int W, int S>
void mixed_block_launch(const Tensor& go, const Tensor& out) {
  constexpr int Rows = S == 1
      ? (W <= 16 ? W : (W >= 56 ? 2 : 4))
      : (W == 14 ? 14 : (W == 28 ? 8 : (W == 64 ? 2 : 4)));
  constexpr int Channels = S == 1 || W == 14 ? 32 : 64;
  using Kernel = AvgPoolMixedBlockTile<scalar_t, W, S, Rows, Channels>;
  const int C = out.size(1), ct = (C + Channels - 1) / Channels,
            yt = (W + Rows - 1) / Rows;
  const int groups = out.size(0) * ct * yt;
  c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> tile(Channels * Kernel::Plane, h);
    h.parallel_for(
        sycl::nd_range<1>(groups * Channels, Channels),
        Kernel{
            reinterpret_cast<const scalar_t*>(go.data_ptr()),
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            C,
            at::detail::IntDivider<uint32_t>(ct),
            at::detail::IntDivider<uint32_t>(yt),
            tile});
  });
}
#endif
template <typename scalar_t, int W, int S, int Rows = 8, int Channels = 32>
struct AvgPoolMixedTile {
  const scalar_t* src;
  scalar_t* dst;
  int C;
  at::detail::IntDivider<uint32_t> channel_tiles, row_tiles;
  sycl::local_accessor<uint32_t, 1> normalized;
  static constexpr int OH = (W + S - 1) / S;
  static constexpr int InputRows = Rows / S + (S == 1 ? 2 : 1);
  static constexpr int Plane = InputRows * OH + 1;
  void operator()(sycl::nd_item<1> it) const {
    const int lid = it.get_local_linear_id();
    const int group = it.get_group_linear_id();
    const int nc = row_tiles.div(group);
    const int y0 = (group - nc * row_tiles.divisor) * Rows;
    const int n = channel_tiles.div(nc);
    const int c0 = (nc - n * channel_tiles.divisor) * Channels;
    const int sy0 = y0 / S - (S == 1 ? 1 : 0);
    for (int i = lid; i < InputRows * OH * Channels / 4; i += 256) {
      const int c = i % (Channels / 4) * 4;
      const int pos = i / (Channels / 4);
      const int sy = sy0 + pos / OH, sx = pos % OH;
      memory::aligned_vector<scalar_t, 4> v{};
      if (sy >= 0 && sy < OH) {
        v = *reinterpret_cast<const decltype(v)*>(
            src + ((n * OH + sy) * OH + sx) * C + c0 + c);
        float inv = 1.f / 9.f;
        if constexpr (S == 1) {
          const int d = (sy == 0 || sy == W - 1 ? 2 : 3) *
              (sx == 0 || sx == W - 1 ? 2 : 3);
          inv = d == 4 ? .25f : d == 6 ? 1.f / 6.f : 1.f / 9.f;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
          v[j] = scalar_t(float(v[j]) * inv);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        memory::aligned_vector<scalar_t, 2> pair{{v[j * 2], v[j * 2 + 1]}};
        normalized[(c / 2 + j) * Plane + pos] = sycl::bit_cast<uint32_t>(pair);
      }
    }
    it.barrier(sycl::access::fence_space::local_space);
    constexpr int OutputPairs = Rows * W / 2;
    for (int i = lid; i < Channels * OutputPairs; i += 256) {
      const int c = i / OutputPairs, pos = i % OutputPairs * 2;
      const int y = y0 + pos / W, x = pos % W;
      if (y < W) {
        memory::aligned_vector<scalar_t, 2> result;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          float sum = 0.f;
          if constexpr (S == 1) {
#pragma unroll
            for (int dy = 0; dy < 3; ++dy) {
#pragma unroll
              for (int dx = -1; dx <= 1; ++dx) {
                const int sx = x + j + dx;
                if (sx >= 0 && sx < W)
                  sum +=
                      float(sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
                          normalized[(c / 2) * Plane + (pos / W + dy) * W + sx])
                                [c % 2]);
              }
            }
          } else {
            const int sy = y / 2 - sy0, sx = (x + j) / 2;
            sum = float(sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
                normalized[(c / 2) * Plane + sy * OH + sx])[c % 2]);
            if ((x + j) % 2 && sx + 1 < OH)
              sum += float(sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
                  normalized[(c / 2) * Plane + sy * OH + sx + 1])[c % 2]);
            if (y % 2 && y / 2 + 1 < OH) {
              sum += float(sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
                  normalized[(c / 2) * Plane + (sy + 1) * OH + sx])[c % 2]);
              if ((x + j) % 2 && sx + 1 < OH)
                sum +=
                    float(sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
                        normalized[(c / 2) * Plane + (sy + 1) * OH + sx + 1])
                              [c % 2]);
            }
          }
          result[j] = scalar_t(sum);
        }
        *reinterpret_cast<decltype(result)*>(
            dst + ((n * C + c0 + c) * W + y) * W + x) = result;
      }
    }
  }
};
template <typename scalar_t, int W, int S>
void mixed_tile_launch(const Tensor& go, const Tensor& out) {
#ifdef USE_SYCLTLA
  if (reinterpret_cast<uintptr_t>(go.data_ptr()) % 64 == 0 &&
      can_use_tla_block_io()) {
    mixed_block_launch<scalar_t, W, S>(go, out);
    return;
  }
#endif
  using Kernel = AvgPoolMixedTile<scalar_t, W, S>;
  const int C = out.size(1), rows = (W + 7) / 8,
            groups = out.size(0) * (C / 32) * rows;
  c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> normalized(16 * Kernel::Plane, h);
    h.parallel_for(
        sycl::nd_range<1>(groups * 256, 256),
        Kernel{
            reinterpret_cast<const scalar_t*>(go.data_ptr()),
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            C,
            at::detail::IntDivider<uint32_t>(C / 32),
            at::detail::IntDivider<uint32_t>(rows),
            normalized});
  });
}
#ifdef USE_SYCLTLA
template <typename scalar_t>
struct AvgPoolPlanarBlock7 {
  const scalar_t* src;
  scalar_t* dst;
  int pairs;
  sycl::local_accessor<uint32_t, 1> tile;
  template <int Part>
  void compute(sycl::nd_item<1> it) const {
    const int lane = it.get_local_linear_id() % 16;
    const int row = it.get_group_linear_id() * 16;
    uint32_t words[56];
#pragma unroll
    for (int b = 0; b < 7; ++b) {
      memory::aligned_vector<uint32_t, 8> raw;
      using Copy = cute::XE_LOAD_2D_TRANSPOSE<32, 16, 8>;
      auto source = cute::make_tensor(
          cute::make_gmem_ptr(reinterpret_cast<const uint32_t*>(src)),
          cute::make_layout(
              cute::make_shape(196, pairs / 4),
              cute::make_stride(cute::_1{}, 196)));
      cute::Copy_Traits<
          Copy,
          cute::_0,
          cute::_1,
          uint32_t,
          cute::Stride<cute::_0, cute::_0>>
          copy(source);
      copy.template update_payload<32>(
          cute::make_coord(Part * 49 / 8 * 8 + b * 8, row));
      Copy::copy(copy.payload, raw.val);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        words[b * 8 + j] = raw[j];
    }
    scalar_t normalized[98];
#pragma unroll
    for (int p = 0; p < 49; ++p) {
      auto input = sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
          words[p + Part * 49 % 8]);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int pos = (p * 2 + j) % 49, y = pos / 7, x = pos % 7;
        const int d = (y == 0 || y == 6 ? 2 : 3) * (x == 0 || x == 6 ? 2 : 3);
        normalized[p * 2 + j] = scalar_t(
            float(input[j]) *
            (d == 4       ? .25f
                 : d == 6 ? 1.f / 6.f
                          : 1.f / 9.f));
      }
    }
    float horizontal[98];
#pragma unroll
    for (int i = 0; i < 98; ++i) {
      constexpr int W = 7;
      const int x = i % W;
      horizontal[i] = (x > 0 ? float(normalized[i - 1]) : 0.f) +
          float(normalized[i]) + (x + 1 < W ? float(normalized[i + 1]) : 0.f);
    }
#pragma unroll
    for (int p = 0; p < 49; ++p) {
      memory::aligned_vector<scalar_t, 2> result;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int index = p * 2 + j, pos = index % 49, y = pos / 7;
        float sum = (y > 0 ? horizontal[index - 7] : 0.f) + horizontal[index] +
            (y < 6 ? horizontal[index + 7] : 0.f);
        result[j] = scalar_t(sum);
      }
      tile[(lane * 4 + Part) * 51 + p] = sycl::bit_cast<uint32_t>(result);
    }
  }
  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const int lid = it.get_local_linear_id(),
              base = it.get_group_linear_id() * 64;
    switch (lid / 16) {
      case 0:
        compute<0>(it);
        break;
      case 1:
        compute<1>(it);
        break;
      case 2:
        compute<2>(it);
        break;
      case 3:
        compute<3>(it);
        break;
    }
    it.barrier(sycl::access::fence_space::local_space);
    for (int i = lid; i < 64 * 49 / 4; i += 64) {
      const int pos = i * 4;
      if (base * 49 + pos < pairs * 49) {
        memory::aligned_vector<uint32_t, 4> out;
#pragma unroll
        for (int j = 0; j < 4; ++j)
          out[j] = tile[(pos + j) / 49 * 51 + (pos + j) % 49];
        *reinterpret_cast<decltype(out)*>(
            reinterpret_cast<uint32_t*>(dst) + base * 49 + pos) = out;
      }
    }
  }
};
template <typename scalar_t>
void planar_block7_launch(const Tensor& go, const Tensor& out) {
  const int pairs = out.numel() / 98;
  c10::xpu::getCurrentXPUStream().queue().submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> tile(64 * 51, h);
    h.parallel_for(
        sycl::nd_range<1>((pairs + 63) / 64 * 64, 64),
        AvgPoolPlanarBlock7<scalar_t>{
            reinterpret_cast<const scalar_t*>(go.data_ptr()),
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            pairs,
            tile});
  });
}
#endif

template <typename scalar_t>
struct AvgPoolPlanar7 {
  const scalar_t* src;
  scalar_t* dst;
  int total;
  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const auto sg = it.get_sub_group();
    const int lane = sg.get_local_linear_id();
    const int start = (it.get_global_linear_id() / 16) * 256;
    if (start >= total)
      return;
    memory::aligned_vector<scalar_t, 16> input{};
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (start + lane + j * 16 < total)
        input[j] = src[start + lane + j * 16];
    float v[18], h[18];
#pragma unroll
    for (int j = 0; j < 18; ++j) {
      const int index = start + lane + (j - 1) * 16;
      scalar_t value;
      if (j == 0 || j == 17)
        value = index >= 0 && index < total ? src[index] : scalar_t(0);
      else
        value = input[j - 1];
      const int p = index % 49, y = p / 7, x = p % 7;
      const int d = (y == 0 || y == 6 ? 2 : 3) * (x == 0 || x == 6 ? 2 : 3);
      v[j] = float(scalar_t(
          float(value) *
          (d == 4       ? .25f
               : d == 6 ? 1.f / 6.f
                        : 1.f / 9.f)));
    }
#pragma unroll
    for (int j = 0; j < 18; ++j) {
      float left_src = lane == 15 ? (j == 0 ? 0.f : v[j - 1]) : v[j];
      float right_src = lane == 0 ? (j == 17 ? 0.f : v[j + 1]) : v[j];
      float left = sycl::select_from_group(sg, left_src, (lane + 15) % 16);
      float right = sycl::select_from_group(sg, right_src, (lane + 1) % 16);
      const int x = (start + lane + (j - 1) * 16) % 7;
      h[j] = (x > 0 ? left : 0.f) + v[j] + (x < 6 ? right : 0.f);
    }
    memory::aligned_vector<scalar_t, 16> output;
#pragma unroll
    for (int j = 1; j <= 16; ++j) {
      float up = sycl::select_from_group(
          sg, lane >= 9 ? h[j - 1] : h[j], (lane + 9) % 16);
      float down = sycl::select_from_group(
          sg, lane < 7 ? h[j + 1] : h[j], (lane + 7) % 16);
      const int y = (start + lane + (j - 1) * 16) % 49 / 7;
      output[j - 1] =
          scalar_t((y == 0 ? 0.f : up) + h[j] + (y == 6 ? 0.f : down));
    }
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (start + lane + j * 16 < total)
        dst[start + lane + j * 16] = output[j];
  }
};
template <typename scalar_t>
void planar7_launch(const Tensor& go, const Tensor& out) {
#ifdef USE_SYCLTLA
  if (reinterpret_cast<uintptr_t>(go.data_ptr()) % 64 == 0 &&
      can_use_tla_block_io()) {
    planar_block7_launch<scalar_t>(go, out);
    return;
  }
#endif
  const int count = out.numel();
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 2047) / 2048 * 128, 128),
      AvgPoolPlanar7<scalar_t>{
          reinterpret_cast<const scalar_t*>(go.data_ptr()),
          reinterpret_cast<scalar_t*>(out.data_ptr()),
          count});
}
struct ZeroInsert {
  const uint32_t* src;
  uint32_t* dst;
  int CP, H, W, OH, OW, total;
  at::detail::IntDivider<uint32_t> dc, dw, dh;
  void operator()(sycl::nd_item<1> it) const {
    int step = it.get_global_range(0) * 4;
    for (int start = it.get_global_linear_id() * 4; start < total;
         start += step) {
      int pix = dc.div(start), ch = start - pix * CP, row = dw.div(pix),
          x = pix - row * W, n = dh.div(row), y = row - n * H;
      memory::aligned_vector<uint32_t, 4> v{};
      if (ch + 4 <= CP && CP % 4 == 0) {
        if (!(x % 2) && !(y % 2))
          v = *reinterpret_cast<const decltype(v)*>(
              src + ((n * OH + y / 2) * OW + x / 2) * CP + ch);
      } else {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int cc = ch + j, xx = x, yy = y, nn = n;
          if (cc >= CP) {
            cc -= CP;
            if (++xx == W) {
              xx = 0;
              if (++yy == H) {
                yy = 0;
                ++nn;
              }
            }
          }
          if (start + j < total && !(xx % 2) && !(yy % 2))
            v[j] = src[((nn * OH + yy / 2) * OW + xx / 2) * CP + cc];
        }
      }
      if (start + 4 <= total)
        *reinterpret_cast<decltype(v)*>(dst + start) = v;
      else {
#pragma unroll
        for (int j = 0; j < 4; j++)
          if (start + j < total)
            dst[start + j] = v[j];
      }
    }
  }
};
struct ZeroInsertSg16 : ZeroInsert {
  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    ZeroInsert::operator()(it);
  }
};
void zero_launch(at::Tensor go, at::Tensor out, int groups) {
  int C = out.size(1), H = out.size(2), W = out.size(3),
      total = out.numel() / 2;
  groups = std::min(groups, (total + 1023) / 1024);
  if (C == 1080 && H == 42 && W == 42) {
    int groups16 = std::min(5120, (total + 511) / 512);
    c10::xpu::getCurrentXPUStream().queue().parallel_for(
        sycl::nd_range<1>(groups16 * 128, 128),
        ZeroInsertSg16{ZeroInsert{
            reinterpret_cast<const uint32_t*>(go.data_ptr()),
            reinterpret_cast<uint32_t*>(out.data_ptr()),
            C / 2,
            H,
            W,
            int(go.size(2)),
            int(go.size(3)),
            total,
            at::detail::IntDivider<uint32_t>(C / 2),
            at::detail::IntDivider<uint32_t>(W),
            at::detail::IntDivider<uint32_t>(H)}});
    return;
  }
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>(groups * 256, 256),
      ZeroInsert{
          reinterpret_cast<const uint32_t*>(go.data_ptr()),
          reinterpret_cast<uint32_t*>(out.data_ptr()),
          C / 2,
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          total,
          at::detail::IntDivider<uint32_t>(C / 2),
          at::detail::IntDivider<uint32_t>(W),
          at::detail::IntDivider<uint32_t>(H)});
}

template <typename scalar_t, int K, int S, int P, bool Include>
void merged_launch(const Tensor& go, const Tensor& out) {
  const int C = out.size(1), W = out.size(3);
  if (out.stride(1) != 1) {
    if (go.stride(1) != 1) {
      if constexpr (K == 3 && S == 1 && !Include)
        planar7_launch<scalar_t>(go, out);
      else
        planar_launch<scalar_t, K, S, P, Include>(go, out);
    } else {
      switch (W) {
        case 14:
          mixed_tile_launch<scalar_t, 14, S>(go, out);
          break;
        case 16:
          mixed_tile_launch<scalar_t, 16, S>(go, out);
          break;
        case 28:
          mixed_tile_launch<scalar_t, 28, S>(go, out);
          break;
        case 32:
          mixed_tile_launch<scalar_t, 32, S>(go, out);
          break;
        case 56:
          mixed_tile_launch<scalar_t, 56, S>(go, out);
          break;
        case 64:
          mixed_tile_launch<scalar_t, 64, S>(go, out);
          break;
        default:
          TORCH_INTERNAL_ASSERT(false);
      }
    }
    return;
  }
  if constexpr (K == 3 && S == 2 && Include) {
    if (C % 32 != 0 && go.stride(3) % 4 == 0) {
      if (C % 8 == 0 && go.stride(3) % 8 == 0)
        stride_two_launch<scalar_t, 8>(go, out);
      else
        stride_two_launch<scalar_t, 4>(go, out);
      return;
    }
  }
  if constexpr (K == 3 && S == 1 && Include) {
    if (go.stride(3) == C && C % 32 == 0 && W <= 17) {
      wide_launch<scalar_t, K, S, P, Include>(go, out);
      return;
    }
  }
  if constexpr (K == 1) {
    if (C % 32 != 0) {
      zero_launch(go, out, 2560);
      return;
    }
  }
  if (C % 8 != 0 || go.stride(3) % 8 != 0) {
    packed_launch<scalar_t, K, S, P, Include>(go, out);
    return;
  }
  launch<scalar_t, K, S, P, Include, 8>(go, out);
}
template <typename scalar_t>
void dispatch_backward(
    const Tensor& go,
    const Tensor& out,
    bool pool3,
    bool pool2,
    int sh,
    bool include) {
  if (pool3 && sh == 1) {
    if (include)
      merged_launch<scalar_t, 3, 1, 1, true>(go, out);
    else
      merged_launch<scalar_t, 3, 1, 1, false>(go, out);
  } else if (pool3) {
    merged_launch<scalar_t, 3, 2, 1, true>(go, out);
  } else if (pool2) {
    if (include)
      merged_launch<scalar_t, 2, 2, 0, true>(go, out);
    else
      merged_launch<scalar_t, 2, 2, 0, false>(go, out);
  } else {
    merged_launch<scalar_t, 1, 2, 0, false>(go, out);
  }
}
inline bool try_backward(
    const Tensor& go,
    const Tensor& input,
    const Tensor& out,
    int kh,
    int kw,
    int sh,
    int sw,
    int ph,
    int pw,
    bool ceil_mode,
    bool include,
    std::optional<int64_t> divisor) {
  if (input.dim() != 4 || go.dim() != 4 || out.dim() != 4 ||
      (input.scalar_type() != kHalf && input.scalar_type() != kBFloat16) ||
      go.scalar_type() != input.scalar_type() ||
      out.scalar_type() != input.scalar_type() || divisor ||
      input.numel() == 0 || kh != kw || sh != sw || ph != pw ||
      !canUse32BitIndexMath(input) || !canUse32BitIndexMath(go) ||
      !canUse32BitIndexMath(out) || out.sizes() != input.sizes()) {
    return false;
  }
  const int C = input.size(1), H = input.size(2), W = input.size(3);
  const bool cl = input.suggest_memory_format() == MemoryFormat::ChannelsLast;
  if (!out.is_contiguous(
          cl ? MemoryFormat::ChannelsLast : MemoryFormat::Contiguous) ||
      reinterpret_cast<uintptr_t>(go.data_ptr()) % 16 ||
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 16) {
    return false;
  }
  const bool pool3 =
      kh == 3 && ph == 1 && !ceil_mode && (sh == 1 || (sh == 2 && include));
  const bool pool2 = kh == 2 && ph == 0 && sh == 2 && H % 2 == 0 && W % 2 == 0;
  const bool pool1 = kh == 1 && ph == 0 && sh == 2 && !ceil_mode;
  if (!pool3 && !pool2 && !pool1) {
    return false;
  }
  const bool go_cl = go.stride(1) == 1 && go.stride(3) >= C &&
      go.stride(3) % 2 == 0 && go.stride(2) == go.size(3) * go.stride(3) &&
      go.stride(0) == go.size(2) * go.stride(2);
  if (cl) {
    if (!go_cl || C < 8 || C % 2 || out.numel() % 8 ||
        (pool3 && sh == 2 && C % 4) ||
        (pool1 && !go.is_contiguous(MemoryFormat::ChannelsLast))) {
      return false;
    }
    if (pool3 && sh == 2 && H == 28 && W == 28 && C == 104 &&
        go.stride(3) % 8) {
      return false;
    }
    if (pool3 && sh == 2 && H == 14 && W == 14 && (C == 112 || C == 208) &&
        go.stride(3) % 8) {
      return false;
    }
  } else if (go.is_contiguous()) {
    if (!(pool3 && sh == 1 && !include && H == 7 && W == 7 &&
          (input.size(0) * C) % 8 == 0) &&
        !(pool2 && W % 8 == 0)) {
      return false;
    }
  } else {
    if (!go.is_contiguous(MemoryFormat::ChannelsLast) || H != W || C % 32 ||
        reinterpret_cast<uintptr_t>(go.data_ptr()) % 64 ||
        !(pool3 &&
          ((sh == 1 && !include && (W == 14 || W == 28 || W == 56)) ||
           (sh == 2 &&
            (W == 14 || W == 16 || W == 28 || W == 32 || W == 56 ||
             W == 64))))) {
      return false;
    }
  }
  const auto device = c10::xpu::getCurrentXPUStream().queue().get_device();
  const auto subgroup_sizes =
      device.get_info<sycl::info::device::sub_group_sizes>();
  // The largest supported portable mixed tile is W=56, S=1: 35904 bytes.
  constexpr size_t local_memory_bytes =
      16 * AvgPoolMixedTile<sycl::half, 56, 1>::Plane * sizeof(uint32_t);
  if (device.get_info<sycl::info::device::max_work_group_size>() < 256 ||
      device.get_info<sycl::info::device::local_mem_size>() <
          local_memory_bytes ||
      std::find(subgroup_sizes.begin(), subgroup_sizes.end(), 16) ==
          subgroup_sizes.end()) {
    return false;
  }
  if (input.scalar_type() == kHalf)
    dispatch_backward<sycl::half>(go, out, pool3, pool2, sh, include);
  else
    dispatch_backward<c10::BFloat16>(go, out, pool3, pool2, sh, include);
  return true;
}
} // namespace avg_pool_backward

// Grid-stride cap for the vectorized channels-last kernels (VEC/WTILE/HWTILE).
// Bounding the launch to a large-but-finite multiple of the device's resident
// thread capacity keeps the grid size sane for huge inputs while preserving
// the oversubscription memory-bound kernels need to hide DRAM latency: a cap
// of 1x (no oversubscription, matching the scalar fallback kernels' cap)
// regresses small-spatial/large-batch shapes by up to 1.86x in benchmarking;
// 32x recovers that (and is within noise of fully uncapped on all shapes
// tested) while still bounding the launch instead of scaling unbounded with
// input size.
static int64_t vectorized_grid_stride_cap() {
  return static_cast<int64_t>(at::xpu::getDeviceMaxWorkItems()) * 32;
}

template <typename scalar_t, typename accscalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_kernel_impl(
    scalar_t* top_data,
    const scalar_t* bottom_data,
    const int total_elements,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, total_elements) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = sycl::min(hstart + kernel_h, static_cast<int>(height + pad_h));
    int wend = sycl::min(wstart + kernel_w, static_cast<int>(width + pad_w));
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = sycl::max(hstart, 0);
    wstart = sycl::max(wstart, 0);
    hend = sycl::min(hend, static_cast<int>(height));
    wend = sycl::min(wend, static_cast<int>(width));

    if (hstart >= hend || wstart >= wend) {
      top_data[index] = scalar_t(0);
      return;
    }

    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_kernel_impl(
    scalar_t* top_data,
    const scalar_t* bottom_data,
    index_t total_elements,
    index_t channels,
    index_t height,
    index_t width,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor,
    at::detail::IntDivider<unsigned int> div_channels,
    at::detail::IntDivider<unsigned int> div_pooled_width,
    at::detail::IntDivider<unsigned int> div_pooled_height) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, total_elements) {
    // Magic-number division replaces chained %/`/` for the index
    // decomposition below.
    auto dm_c = div_channels.divmod(index);
    const int c = dm_c.mod;
    auto dm_w = div_pooled_width.divmod(dm_c.div);
    const int pw = dm_w.mod;
    auto dm_h = div_pooled_height.divmod(dm_w.div);
    const int ph = dm_h.mod;
    const int n = dm_h.div;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = sycl::min(hstart + kernel_h, static_cast<int>(height + pad_h));
    int wend = sycl::min(wstart + kernel_w, static_cast<int>(width + pad_w));
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = sycl::max(hstart, 0);
    wstart = sycl::max(wstart, 0);
    hend = sycl::min(hend, static_cast<int>(height));
    wend = sycl::min(wend, static_cast<int>(width));

    if (hstart >= hend || wstart >= wend) {
      top_data[index] = scalar_t(0);
      return;
    }

    accscalar_t aveval = accscalar_t(0);
    const scalar_t* const bottom_slice =
        bottom_data + n * channels * height * width + c;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[(h * width + w) * channels];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = static_cast<scalar_t>(aveval / divide_factor);
  }
}

// Each work-item processes vec_size channels at once using aligned_vector
// loads.
template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    int vec_size,
    typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_vec_kernel_impl(
    vec_t* output_vec,
    const vec_t* input_vec,
    index_t total_elements,
    index_t num_vec_channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor,
    at::detail::IntDivider<unsigned int> div_num_vec_channels,
    at::detail::IntDivider<unsigned int> div_pooled_width,
    at::detail::IntDivider<unsigned int> div_pooled_height,
    index_t nvc_mask,
    int nvc_shift) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  // total_elements = N * oH * oW * (C / vec_size)
  for (index_t idx = item.get_global_linear_id(); idx < total_elements;
       idx += item.get_local_range(0) * item.get_group_range(0)) {
    // Decompose: idx -> (plane, pw, ph, n)
    // Use bitshift for power-of-2 num_vec_channels, else IntDivider
    index_t plane, spatial_idx;
    if (nvc_shift > 0) {
      plane = idx & nvc_mask;
      spatial_idx = idx >> nvc_shift;
    } else {
      auto dm_c = div_num_vec_channels.divmod(idx);
      plane = dm_c.mod;
      spatial_idx = dm_c.div;
    }
    auto dm_w = div_pooled_width.divmod(spatial_idx);
    const index_t pw = dm_w.mod;
    auto dm_h = div_pooled_height.divmod(dm_w.div);
    const index_t ph = dm_h.mod;
    const index_t n = dm_h.div;

    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = sycl::min(hstart + kernel_h, height + pad_h);
    int wend = sycl::min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = sycl::max(hstart, 0);
    wstart = sycl::max(wstart, 0);
    hend = sycl::min(hend, height);
    wend = sycl::min(wend, width);

    if (hstart >= hend || wstart >= wend) {
      vec_t zero_vec;
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        zero_vec.val[i] = scalar_t(0);
      output_vec[idx] = zero_vec;
      continue;
    }

    accscalar_t aveval[vec_size];

    // input layout: N, H, W, C (channels-last)
    // input_vec layout: N, H, W, C/vec_size
    const index_t batch_offset = n * height * width * num_vec_channels;
    const index_t row_stride = width * num_vec_channels;

    // Fast path: interior pixels with full 3x3 window.
    // All 9 loads issued upfront so HW memory pipeline can overlap them.
    if (kernel_h == 3 && kernel_w == 3 && (hend - hstart) == 3 &&
        (wend - wstart) == 3) {
      const index_t base = batch_offset + hstart * row_stride +
          wstart * num_vec_channels + plane;
      vec_t v00 = input_vec[base];
      vec_t v01 = input_vec[base + num_vec_channels];
      vec_t v02 = input_vec[base + 2 * num_vec_channels];
      vec_t v10 = input_vec[base + row_stride];
      vec_t v11 = input_vec[base + row_stride + num_vec_channels];
      vec_t v12 = input_vec[base + row_stride + 2 * num_vec_channels];
      vec_t v20 = input_vec[base + 2 * row_stride];
      vec_t v21 = input_vec[base + 2 * row_stride + num_vec_channels];
      vec_t v22 = input_vec[base + 2 * row_stride + 2 * num_vec_channels];
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        aveval[i] = static_cast<accscalar_t>(v00.val[i]) +
            static_cast<accscalar_t>(v01.val[i]) +
            static_cast<accscalar_t>(v02.val[i]) +
            static_cast<accscalar_t>(v10.val[i]) +
            static_cast<accscalar_t>(v11.val[i]) +
            static_cast<accscalar_t>(v12.val[i]) +
            static_cast<accscalar_t>(v20.val[i]) +
            static_cast<accscalar_t>(v21.val[i]) +
            static_cast<accscalar_t>(v22.val[i]);
      }
    } else if (
        kernel_h == 2 && kernel_w == 2 && (hend - hstart) == 2 &&
        (wend - wstart) == 2) {
      // Fast path: full 2x2 window (k2s2 interior pixels).
      // All 4 loads issued upfront for memory pipeline overlap.
      const index_t base = batch_offset + hstart * row_stride +
          wstart * num_vec_channels + plane;
      vec_t v00 = input_vec[base];
      vec_t v01 = input_vec[base + num_vec_channels];
      vec_t v10 = input_vec[base + row_stride];
      vec_t v11 = input_vec[base + row_stride + num_vec_channels];
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        aveval[i] = static_cast<accscalar_t>(v00.val[i]) +
            static_cast<accscalar_t>(v01.val[i]) +
            static_cast<accscalar_t>(v10.val[i]) +
            static_cast<accscalar_t>(v11.val[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        aveval[i] = accscalar_t(0);
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          index_t load_idx =
              batch_offset + (h * width + w) * num_vec_channels + plane;
          vec_t val = input_vec[load_idx];
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            aveval[i] += static_cast<accscalar_t>(val.val[i]);
          }
        }
      }
    }

    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if (count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }

    vec_t out_vec;
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      out_vec.val[i] = static_cast<scalar_t>(aveval[i] / divide_factor);
    }
    output_vec[idx] = out_vec;
  }
}

// W-tile kernel: each work-item processes 2 adjacent output positions in W
// direction, sharing overlapping input columns for k=3 s=1.
// Reduces total loads by 33% (12 loads for 2 outputs vs 18 independently).
template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    int vec_size,
    typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_vec_wtile_kernel_impl(
    vec_t* output_vec,
    const vec_t* input_vec,
    index_t total_elements,
    index_t num_vec_channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int stride_h,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor,
    at::detail::IntDivider<unsigned int> div_num_vec_channels,
    at::detail::IntDivider<unsigned int> div_pooled_width_half,
    at::detail::IntDivider<unsigned int> div_pooled_height,
    index_t nvc_mask,
    int nvc_shift) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  // total_elements = N * oH * ceil(oW/2) * num_vec_channels
  for (index_t idx = item.get_global_linear_id(); idx < total_elements;
       idx += item.get_local_range(0) * item.get_group_range(0)) {
    // Index decomposition: idx -> (plane, pw_pair, ph, n)
    index_t plane, spatial_idx;
    if (nvc_shift > 0) {
      plane = idx & nvc_mask;
      spatial_idx = idx >> nvc_shift;
    } else {
      auto dm_c = div_num_vec_channels.divmod(idx);
      plane = dm_c.mod;
      spatial_idx = dm_c.div;
    }
    auto dm_w = div_pooled_width_half.divmod(spatial_idx);
    const index_t pw_pair = dm_w.mod;
    auto dm_h = div_pooled_height.divmod(dm_w.div);
    const index_t ph = dm_h.mod;
    const index_t n = dm_h.div;

    const int pw0 = pw_pair * 2;
    const int pw1 = pw0 + 1;

    // Compute H bounds (same for both outputs)
    int hstart = ph * stride_h - pad_h;
    int hend = sycl::min(hstart + 3, height + pad_h);
    const int pool_h_full = hend - hstart;
    hstart = sycl::max(hstart, 0);
    hend = sycl::min(hend, height);

    // Compute W bounds for pw0
    int wstart0 = pw0 - pad_w;
    int wend0 = sycl::min(wstart0 + 3, width + pad_w);
    const int pool_w0_full = wend0 - wstart0;
    wstart0 = sycl::max(wstart0, 0);
    wend0 = sycl::min(wend0, width);

    // Compute W bounds for pw1
    int wstart1 = pw1 - pad_w;
    int wend1 = sycl::min(wstart1 + 3, width + pad_w);
    const int pool_w1_full = wend1 - wstart1;
    wstart1 = sycl::max(wstart1, 0);
    wend1 = sycl::min(wend1, width);

    const index_t batch_offset = n * height * width * num_vec_channels;

    accscalar_t acc0[vec_size], acc1[vec_size];

    // Fast path: both outputs have full 3x3 windows (interior pixels).
    // Unroll all 12 loads (3 rows x 4 cols) upfront.
    if ((hend - hstart) == 3 && (wend0 - wstart0) == 3 &&
        (wend1 - wstart1) == 3) {
      const index_t row_stride = width * num_vec_channels;
      const index_t base = batch_offset + hstart * row_stride +
          wstart0 * num_vec_channels + plane;
      // 3 rows, 4 columns: wstart0, wstart0+1, wstart0+2, wstart0+3
      // wstart0+0 = unique to pw0
      // wstart0+1, wstart0+2 = shared (pw0 and pw1)
      // wstart0+3 = unique to pw1
      vec_t r0c0 = input_vec[base];
      vec_t r0c1 = input_vec[base + num_vec_channels];
      vec_t r0c2 = input_vec[base + 2 * num_vec_channels];
      vec_t r0c3 = input_vec[base + 3 * num_vec_channels];
      vec_t r1c0 = input_vec[base + row_stride];
      vec_t r1c1 = input_vec[base + row_stride + num_vec_channels];
      vec_t r1c2 = input_vec[base + row_stride + 2 * num_vec_channels];
      vec_t r1c3 = input_vec[base + row_stride + 3 * num_vec_channels];
      vec_t r2c0 = input_vec[base + 2 * row_stride];
      vec_t r2c1 = input_vec[base + 2 * row_stride + num_vec_channels];
      vec_t r2c2 = input_vec[base + 2 * row_stride + 2 * num_vec_channels];
      vec_t r2c3 = input_vec[base + 2 * row_stride + 3 * num_vec_channels];
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        accscalar_t c0 = static_cast<accscalar_t>(r0c0.val[i]) +
            static_cast<accscalar_t>(r1c0.val[i]) +
            static_cast<accscalar_t>(r2c0.val[i]);
        accscalar_t c1 = static_cast<accscalar_t>(r0c1.val[i]) +
            static_cast<accscalar_t>(r1c1.val[i]) +
            static_cast<accscalar_t>(r2c1.val[i]);
        accscalar_t c2 = static_cast<accscalar_t>(r0c2.val[i]) +
            static_cast<accscalar_t>(r1c2.val[i]) +
            static_cast<accscalar_t>(r2c2.val[i]);
        accscalar_t c3 = static_cast<accscalar_t>(r0c3.val[i]) +
            static_cast<accscalar_t>(r1c3.val[i]) +
            static_cast<accscalar_t>(r2c3.val[i]);
        acc0[i] = c0 + c1 + c2; // cols 0,1,2
        acc1[i] = c1 + c2 + c3; // cols 1,2,3
      }
    } else {
      // Generic path for edge pixels
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        acc0[i] = accscalar_t(0);
        acc1[i] = accscalar_t(0);
      }

      // Shared and unique column ranges for data reuse
      const int shared_wstart = sycl::max(wstart0, wstart1);
      const int shared_wend = sycl::min(wend0, wend1);

      for (int h = hstart; h < hend; ++h) {
        const index_t row_base = batch_offset + h * width * num_vec_channels;

        // Columns unique to pw0 (left of shared region)
        for (int w = wstart0; w < shared_wstart; ++w) {
          vec_t val = input_vec[row_base + w * num_vec_channels + plane];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc0[i] += static_cast<accscalar_t>(val.val[i]);
        }

        // Shared columns (add to both accumulators)
        for (int w = shared_wstart; w < shared_wend; ++w) {
          vec_t val = input_vec[row_base + w * num_vec_channels + plane];
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            acc0[i] += static_cast<accscalar_t>(val.val[i]);
            acc1[i] += static_cast<accscalar_t>(val.val[i]);
          }
        }

        // Columns unique to pw1 (right of shared region)
        for (int w = shared_wend; w < wend1; ++w) {
          vec_t val = input_vec[row_base + w * num_vec_channels + plane];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc1[i] += static_cast<accscalar_t>(val.val[i]);
        }
      }
    }

    // Compute divide factors
    int divide0, divide1;
    if (use_divisor) {
      divide0 = divide1 = divisor_override;
    } else if (count_include_pad) {
      divide0 = pool_h_full * pool_w0_full;
      divide1 = pool_h_full * pool_w1_full;
    } else {
      divide0 = (hend - hstart) * (wend0 - wstart0);
      divide1 = (hend - hstart) * (wend1 - wstart1);
    }

    // Write output for pw0
    const index_t out_idx0 =
        n * pooled_height * pooled_width * num_vec_channels +
        ph * pooled_width * num_vec_channels + pw0 * num_vec_channels + plane;
    vec_t out0;
#pragma unroll
    for (int i = 0; i < vec_size; i++)
      out0.val[i] = static_cast<scalar_t>(acc0[i] / divide0);
    output_vec[out_idx0] = out0;

    // Write output for pw1 (bounds check for odd pooled_width)
    if (pw1 < pooled_width) {
      const index_t out_idx1 = out_idx0 + num_vec_channels;
      vec_t out1;
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        out1.val[i] = static_cast<scalar_t>(acc1[i] / divide1);
      output_vec[out_idx1] = out1;
    }
  }
}

// HW-tile kernel: each work-item processes a 2x2 block of adjacent outputs
// (ph0,pw0), (ph0,pw1), (ph1,pw0), (ph1,pw1) for k=3 s=1.
// Loads a 4x4 input grid (16 loads) shared across 4 outputs = 4 loads/output
// vs 9 in flat. Used only for vec_size==2 shapes where WTile subgroup
// alignment is poor (num_vc not a multiple of subgroup width).
template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    int vec_size,
    typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_vec_hwtile_kernel_impl(
    vec_t* output_vec,
    const vec_t* input_vec,
    index_t total_elements,
    index_t num_vec_channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int stride_h,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor,
    at::detail::IntDivider<unsigned int> div_num_vec_channels,
    at::detail::IntDivider<unsigned int> div_pooled_width_half,
    at::detail::IntDivider<unsigned int> div_pooled_height_half,
    index_t nvc_mask,
    int nvc_shift) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  // total_elements = N * ceil(oH/2) * ceil(oW/2) * num_vec_channels
  for (index_t idx = item.get_global_linear_id(); idx < total_elements;
       idx += item.get_local_range(0) * item.get_group_range(0)) {
    index_t plane, spatial_idx;
    if (nvc_shift > 0) {
      plane = idx & nvc_mask;
      spatial_idx = idx >> nvc_shift;
    } else {
      auto dm_c = div_num_vec_channels.divmod(idx);
      plane = dm_c.mod;
      spatial_idx = dm_c.div;
    }
    auto dm_w = div_pooled_width_half.divmod(spatial_idx);
    const index_t pw_pair = dm_w.mod;
    auto dm_h = div_pooled_height_half.divmod(dm_w.div);
    const index_t ph_pair = dm_h.mod;
    const index_t n = dm_h.div;

    const int pw0 = pw_pair * 2;
    const int pw1 = pw0 + 1;
    const int ph0 = ph_pair * 2;
    const int ph1 = ph0 + 1;

    int hstart0 = ph0 * stride_h - pad_h;
    int hend0 = sycl::min(hstart0 + 3, height + pad_h);
    const int pool_h0_full = hend0 - hstart0;
    hstart0 = sycl::max(hstart0, 0);
    hend0 = sycl::min(hend0, height);

    int hstart1 = ph1 * stride_h - pad_h;
    int hend1 = sycl::min(hstart1 + 3, height + pad_h);
    const int pool_h1_full = hend1 - hstart1;
    hstart1 = sycl::max(hstart1, 0);
    hend1 = sycl::min(hend1, height);

    int wstart0 = pw0 - pad_w;
    int wend0 = sycl::min(wstart0 + 3, width + pad_w);
    const int pool_w0_full = wend0 - wstart0;
    wstart0 = sycl::max(wstart0, 0);
    wend0 = sycl::min(wend0, width);

    int wstart1 = pw1 - pad_w;
    int wend1 = sycl::min(wstart1 + 3, width + pad_w);
    const int pool_w1_full = wend1 - wstart1;
    wstart1 = sycl::max(wstart1, 0);
    wend1 = sycl::min(wend1, width);

    const index_t batch_offset = n * height * width * num_vec_channels;

    accscalar_t acc00[vec_size], acc01[vec_size];
    accscalar_t acc10[vec_size], acc11[vec_size];

    // Fast path: all 4 outputs are interior (full 3x3 window, ph1 valid).
    // Load a 4x4 grid (16 loads), then combine with column partial sums.
    if (ph1 < pooled_height && (hend0 - hstart0) == 3 &&
        (hend1 - hstart1) == 3 && (wend0 - wstart0) == 3 &&
        (wend1 - wstart1) == 3) {
      const index_t row_stride = width * num_vec_channels;
      const index_t base = batch_offset + hstart0 * row_stride +
          wstart0 * num_vec_channels + plane;
      vec_t r0c0 = input_vec[base];
      vec_t r0c1 = input_vec[base + num_vec_channels];
      vec_t r0c2 = input_vec[base + 2 * num_vec_channels];
      vec_t r0c3 = input_vec[base + 3 * num_vec_channels];
      vec_t r1c0 = input_vec[base + row_stride];
      vec_t r1c1 = input_vec[base + row_stride + num_vec_channels];
      vec_t r1c2 = input_vec[base + row_stride + 2 * num_vec_channels];
      vec_t r1c3 = input_vec[base + row_stride + 3 * num_vec_channels];
      vec_t r2c0 = input_vec[base + 2 * row_stride];
      vec_t r2c1 = input_vec[base + 2 * row_stride + num_vec_channels];
      vec_t r2c2 = input_vec[base + 2 * row_stride + 2 * num_vec_channels];
      vec_t r2c3 = input_vec[base + 2 * row_stride + 3 * num_vec_channels];
      vec_t r3c0 = input_vec[base + 3 * row_stride];
      vec_t r3c1 = input_vec[base + 3 * row_stride + num_vec_channels];
      vec_t r3c2 = input_vec[base + 3 * row_stride + 2 * num_vec_channels];
      vec_t r3c3 = input_vec[base + 3 * row_stride + 3 * num_vec_channels];
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        // Column sums for rows 0..2 (used by ph0 outputs)
        accscalar_t c0_012 = static_cast<accscalar_t>(r0c0.val[i]) +
            static_cast<accscalar_t>(r1c0.val[i]) +
            static_cast<accscalar_t>(r2c0.val[i]);
        accscalar_t c1_012 = static_cast<accscalar_t>(r0c1.val[i]) +
            static_cast<accscalar_t>(r1c1.val[i]) +
            static_cast<accscalar_t>(r2c1.val[i]);
        accscalar_t c2_012 = static_cast<accscalar_t>(r0c2.val[i]) +
            static_cast<accscalar_t>(r1c2.val[i]) +
            static_cast<accscalar_t>(r2c2.val[i]);
        accscalar_t c3_012 = static_cast<accscalar_t>(r0c3.val[i]) +
            static_cast<accscalar_t>(r1c3.val[i]) +
            static_cast<accscalar_t>(r2c3.val[i]);
        // Column sums for rows 1..3 (used by ph1 outputs)
        accscalar_t c0_123 = static_cast<accscalar_t>(r1c0.val[i]) +
            static_cast<accscalar_t>(r2c0.val[i]) +
            static_cast<accscalar_t>(r3c0.val[i]);
        accscalar_t c1_123 = static_cast<accscalar_t>(r1c1.val[i]) +
            static_cast<accscalar_t>(r2c1.val[i]) +
            static_cast<accscalar_t>(r3c1.val[i]);
        accscalar_t c2_123 = static_cast<accscalar_t>(r1c2.val[i]) +
            static_cast<accscalar_t>(r2c2.val[i]) +
            static_cast<accscalar_t>(r3c2.val[i]);
        accscalar_t c3_123 = static_cast<accscalar_t>(r1c3.val[i]) +
            static_cast<accscalar_t>(r2c3.val[i]) +
            static_cast<accscalar_t>(r3c3.val[i]);
        acc00[i] = c0_012 + c1_012 + c2_012; // ph0,pw0: rows 0..2, cols 0..2
        acc01[i] = c1_012 + c2_012 + c3_012; // ph0,pw1: rows 0..2, cols 1..3
        acc10[i] = c0_123 + c1_123 + c2_123; // ph1,pw0: rows 1..3, cols 0..2
        acc11[i] = c1_123 + c2_123 + c3_123; // ph1,pw1: rows 1..3, cols 1..3
      }
    } else {
      // Generic slow path: each output computed independently.
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        acc00[i] = acc01[i] = acc10[i] = acc11[i] = accscalar_t(0);

      for (int h = hstart0; h < hend0; ++h) {
        const index_t rbase =
            batch_offset + h * width * num_vec_channels + plane;
        for (int w = wstart0; w < wend0; ++w) {
          vec_t v = input_vec[rbase + w * num_vec_channels];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc00[i] += static_cast<accscalar_t>(v.val[i]);
        }
        for (int w = wstart1; w < wend1; ++w) {
          vec_t v = input_vec[rbase + w * num_vec_channels];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc01[i] += static_cast<accscalar_t>(v.val[i]);
        }
      }
      for (int h = hstart1; h < hend1; ++h) {
        const index_t rbase =
            batch_offset + h * width * num_vec_channels + plane;
        for (int w = wstart0; w < wend0; ++w) {
          vec_t v = input_vec[rbase + w * num_vec_channels];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc10[i] += static_cast<accscalar_t>(v.val[i]);
        }
        for (int w = wstart1; w < wend1; ++w) {
          vec_t v = input_vec[rbase + w * num_vec_channels];
#pragma unroll
          for (int i = 0; i < vec_size; i++)
            acc11[i] += static_cast<accscalar_t>(v.val[i]);
        }
      }
    }

    // Divide factors for ph0 outputs
    int div00, div01;
    if (use_divisor) {
      div00 = div01 = divisor_override;
    } else if (count_include_pad) {
      div00 = pool_h0_full * pool_w0_full;
      div01 = pool_h0_full * pool_w1_full;
    } else {
      div00 = (hend0 - hstart0) * (wend0 - wstart0);
      div01 = (hend0 - hstart0) * (wend1 - wstart1);
    }

    const index_t out_base =
        n * pooled_height * pooled_width * num_vec_channels +
        ph0 * pooled_width * num_vec_channels;
    const index_t out00 = out_base + pw0 * num_vec_channels + plane;

    vec_t o00;
#pragma unroll
    for (int i = 0; i < vec_size; i++)
      o00.val[i] = static_cast<scalar_t>(acc00[i] / div00);
    output_vec[out00] = o00;

    if (pw1 < pooled_width) {
      vec_t o01;
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        o01.val[i] = static_cast<scalar_t>(acc01[i] / div01);
      output_vec[out00 + num_vec_channels] = o01;
    }

    // Write ph1 outputs only if ph1 is in bounds
    if (ph1 < pooled_height) {
      int div10, div11;
      if (use_divisor) {
        div10 = div11 = divisor_override;
      } else if (count_include_pad) {
        div10 = pool_h1_full * pool_w0_full;
        div11 = pool_h1_full * pool_w1_full;
      } else {
        div10 = (hend1 - hstart1) * (wend0 - wstart0);
        div11 = (hend1 - hstart1) * (wend1 - wstart1);
      }
      const index_t out10 = out_base + pooled_width * num_vec_channels +
          pw0 * num_vec_channels + plane;
      vec_t o10;
#pragma unroll
      for (int i = 0; i < vec_size; i++)
        o10.val[i] = static_cast<scalar_t>(acc10[i] / div10);
      output_vec[out10] = o10;

      if (pw1 < pooled_width) {
        vec_t o11;
#pragma unroll
        for (int i = 0; i < vec_size; i++)
          o11.val[i] = static_cast<scalar_t>(acc11[i] / div11);
        output_vec[out10 + num_vec_channels] = o11;
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_channels_last_kernel(
    const int total_elements,
    const Tensor& input,
    const index_t channels,
    const index_t height,
    const index_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& output,
    const int divisor_override,
    const bool count_include_pad,
    const bool use_divisor) {
  scalar_t* top_data = output.mutable_data_ptr<scalar_t>();
  const scalar_t* bottom_data = input.const_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const int64_t group_size = static_cast<int64_t>(
      at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore());

  // Vectorized path: pick largest vec_size that (1) divides channels,
  // (2) maintains enough work-items for good EU occupancy, and
  // (3) passes alignment check.
  const index_t spatial_out =
      static_cast<index_t>(pooled_height) * static_cast<index_t>(pooled_width);
  const index_t batch_count = total_elements / (channels * spatial_out);
  // Device-derived occupancy floor instead of a fixed constant, so the
  // threshold scales with the actual GPU's EU/thread capacity. The divisor
  // is chosen to reproduce the empirically-tuned value on BMG exactly
  // (at::xpu::getDeviceMaxWorkItems() == 40960 on BMG, 40960 / 5 == 8192).
  const int64_t min_work_items =
      static_cast<int64_t>(at::xpu::getDeviceMaxWorkItems()) / 5;

  // Determine max vec_size from pointer alignment alone (bypass conservative
  // preferred_vector_width which caps at 4 on BMG). The hardware supports
  // wider loads; the preferred_width is a SYCL hint, not a hard limit.
  int max_vs = 8; // start at 8, we'll validate alignment
  {
    uint64_t addr_in = reinterpret_cast<uint64_t>(bottom_data);
    uint64_t addr_out = reinterpret_cast<uint64_t>(top_data);
    constexpr int align8 = sizeof(scalar_t) * 8; // 32 bytes for float
    constexpr int align4 = sizeof(scalar_t) * 4; // 16 bytes for float
    if (addr_in % align8 != 0 || addr_out % align8 != 0) {
      if (addr_in % align4 == 0 && addr_out % align4 == 0)
        max_vs = 4;
      else
        max_vs = 1;
    }
  }

  // For k=3,s=1 shapes that qualify for W-tile, prefer vec4 so W-tile can
  // activate (W-tile + vec4 is better than plain vec8 for large shapes due
  // to 33% load sharing). Only allow vec8 when W-tile won't be used.
  const bool wtile_eligible =
      (kernel_h == 3 && kernel_w == 3 && stride_w == 1 && pooled_width >= 2);
  if (wtile_eligible && max_vs > 4) {
    // Check if vec4 + W-tile would meet the wtile threshold
    const index_t num_vc4 = channels / 4;
    const index_t pw_half = (static_cast<index_t>(pooled_width) + 1) / 2;
    const index_t wtile_items_v4 =
        batch_count * static_cast<index_t>(pooled_height) * pw_half * num_vc4;
    if (wtile_items_v4 >= 8 * min_work_items) {
      // W-tile would activate with vec4 -> cap at vec4 to get load sharing
      max_vs = 4;
    }
  }

  // Don't allow vec8 if it would reduce work-items below the minimum
  // occupancy threshold.
  if (max_vs > 4 && channels % 8 == 0) {
    index_t vec8_items = batch_count * spatial_out * (channels / 8);
    if (vec8_items < min_work_items) {
      max_vs = 4;
    }
  }

  int vec_size = 1;
  for (int vs = max_vs; vs > 1; vs /= 2) {
    if (channels % vs != 0)
      continue;
    index_t vec_work_items = batch_count * spatial_out * (channels / vs);
    if (vec_work_items >= min_work_items) {
      vec_size = vs;
      break;
    }
  }

  // W-tile path for k=3, stride_w=1: each WI handles 2 adjacent pw outputs
  // with shared input column reuse (33% fewer loads).
  // Only activate when halved work-items still provide enough occupancy.
  const bool use_wtile = [&]() {
    if (kernel_h != 3 || kernel_w != 3 || stride_w != 1 || pooled_width < 2)
      return false;
    const index_t num_vc = channels / vec_size;
    const index_t pooled_width_half =
        (static_cast<index_t>(pooled_width) + 1) / 2;
    const index_t wtile_items = batch_count *
        static_cast<index_t>(pooled_height) * pooled_width_half * num_vc;
    // W-tile halves the work-item count for a smaller per-output saving than
    // a larger vec_size, so it needs more headroom above the occupancy floor
    // to be worth activating (8x reproduces the tuned 65536 on BMG).
    return wtile_items >= 8 * min_work_items;
  }();

  // Bitshift-based division is used in place of IntDivider when
  // num_vec_channels is a power of 2 (cheaper than magic-number divmod).
  auto nvc_shift_for = [](index_t num_vec_channels) {
    int shift = 0;
    if ((num_vec_channels & (num_vec_channels - 1)) == 0) {
      index_t tmp = num_vec_channels;
      while (tmp > 1) {
        shift++;
        tmp >>= 1;
      }
    }
    return shift;
  };

#define LAUNCH_AVG_POOL2D_CL_VEC(VS)                                          \
  {                                                                           \
    using vec_t = memory::aligned_vector<scalar_t, VS>;                       \
    const index_t num_vec_channels = channels / VS;                           \
    const index_t vec_total = (total_elements / channels) * num_vec_channels; \
    const int64_t gr =                                                        \
        xpuKernelLoopGroupRange(                                              \
            vec_total, group_size, vectorized_grid_stride_cap()) *            \
        group_size;                                                           \
    sycl_kernel_submit<avg_pool2d_channels_last_vec_kernel_impl<              \
        scalar_t,                                                             \
        accscalar_t,                                                          \
        vec_t,                                                                \
        VS,                                                                   \
        index_t>>(                                                            \
        gr,                                                                   \
        group_size,                                                           \
        queue,                                                                \
        0,                                                                    \
        reinterpret_cast<vec_t*>(top_data),                                   \
        reinterpret_cast<const vec_t*>(bottom_data),                          \
        vec_total,                                                            \
        num_vec_channels,                                                     \
        static_cast<int>(height),                                             \
        static_cast<int>(width),                                              \
        pooled_height,                                                        \
        pooled_width,                                                         \
        kernel_h,                                                             \
        kernel_w,                                                             \
        stride_h,                                                             \
        stride_w,                                                             \
        pad_h,                                                                \
        pad_w,                                                                \
        divisor_override,                                                     \
        count_include_pad,                                                    \
        use_divisor,                                                          \
        at::detail::IntDivider<unsigned int>(                                 \
            static_cast<unsigned int>(num_vec_channels)),                     \
        at::detail::IntDivider<unsigned int>(                                 \
            static_cast<unsigned int>(pooled_width)),                         \
        at::detail::IntDivider<unsigned int>(                                 \
            static_cast<unsigned int>(pooled_height)),                        \
        static_cast<index_t>(num_vec_channels - 1),                           \
        nvc_shift_for(num_vec_channels));                                     \
  }

#define LAUNCH_AVG_POOL2D_CL_WTILE(VS)                                 \
  {                                                                    \
    using vec_t = memory::aligned_vector<scalar_t, VS>;                \
    const index_t num_vec_channels = channels / VS;                    \
    const index_t pooled_width_half =                                  \
        (static_cast<index_t>(pooled_width) + 1) / 2;                  \
    const index_t wtile_total = batch_count *                          \
        static_cast<index_t>(pooled_height) * pooled_width_half *      \
        num_vec_channels;                                              \
    const int64_t gr =                                                 \
        xpuKernelLoopGroupRange(                                       \
            wtile_total, group_size, vectorized_grid_stride_cap()) *   \
        group_size;                                                    \
    sycl_kernel_submit<avg_pool2d_channels_last_vec_wtile_kernel_impl< \
        scalar_t,                                                      \
        accscalar_t,                                                   \
        vec_t,                                                         \
        VS,                                                            \
        index_t>>(                                                     \
        gr,                                                            \
        group_size,                                                    \
        queue,                                                         \
        0,                                                             \
        reinterpret_cast<vec_t*>(top_data),                            \
        reinterpret_cast<const vec_t*>(bottom_data),                   \
        wtile_total,                                                   \
        num_vec_channels,                                              \
        static_cast<int>(height),                                      \
        static_cast<int>(width),                                       \
        pooled_height,                                                 \
        pooled_width,                                                  \
        stride_h,                                                      \
        pad_h,                                                         \
        pad_w,                                                         \
        divisor_override,                                              \
        count_include_pad,                                             \
        use_divisor,                                                   \
        at::detail::IntDivider<unsigned int>(                          \
            static_cast<unsigned int>(num_vec_channels)),              \
        at::detail::IntDivider<unsigned int>(                          \
            static_cast<unsigned int>(pooled_width_half)),             \
        at::detail::IntDivider<unsigned int>(                          \
            static_cast<unsigned int>(pooled_height)),                 \
        static_cast<index_t>(num_vec_channels - 1),                    \
        nvc_shift_for(num_vec_channels));                              \
  }

#define LAUNCH_AVG_POOL2D_CL_HWTILE(VS)                                 \
  {                                                                     \
    using vec_t = memory::aligned_vector<scalar_t, VS>;                 \
    const index_t num_vec_channels = channels / VS;                     \
    const index_t pooled_width_half =                                   \
        (static_cast<index_t>(pooled_width) + 1) / 2;                   \
    const index_t pooled_height_half =                                  \
        (static_cast<index_t>(pooled_height) + 1) / 2;                  \
    const index_t hwtile_total = batch_count * pooled_height_half *     \
        pooled_width_half * num_vec_channels;                           \
    const int64_t gr =                                                  \
        xpuKernelLoopGroupRange(                                        \
            hwtile_total, group_size, vectorized_grid_stride_cap()) *   \
        group_size;                                                     \
    sycl_kernel_submit<avg_pool2d_channels_last_vec_hwtile_kernel_impl< \
        scalar_t,                                                       \
        accscalar_t,                                                    \
        vec_t,                                                          \
        VS,                                                             \
        index_t>>(                                                      \
        gr,                                                             \
        group_size,                                                     \
        queue,                                                          \
        0,                                                              \
        reinterpret_cast<vec_t*>(top_data),                             \
        reinterpret_cast<const vec_t*>(bottom_data),                    \
        hwtile_total,                                                   \
        num_vec_channels,                                               \
        static_cast<int>(height),                                       \
        static_cast<int>(width),                                        \
        pooled_height,                                                  \
        pooled_width,                                                   \
        stride_h,                                                       \
        pad_h,                                                          \
        pad_w,                                                          \
        divisor_override,                                               \
        count_include_pad,                                              \
        use_divisor,                                                    \
        at::detail::IntDivider<unsigned int>(                           \
            static_cast<unsigned int>(num_vec_channels)),               \
        at::detail::IntDivider<unsigned int>(                           \
            static_cast<unsigned int>(pooled_width_half)),              \
        at::detail::IntDivider<unsigned int>(                           \
            static_cast<unsigned int>(pooled_height_half)),             \
        static_cast<index_t>(num_vec_channels - 1),                     \
        nvc_shift_for(num_vec_channels));                               \
  }

  if (use_wtile) {
    if (vec_size == 2) {
      // HWTILE's fast path reuses one shared 4-row input grid, assuming
      // ph1's window starts exactly 1 row after ph0's (rows 1..3 of the
      // loaded grid). That assumption only holds when stride_h == 1;
      // otherwise fall back to plain VEC to avoid computing wrong sums.
      if (stride_h == 1) {
        LAUNCH_AVG_POOL2D_CL_HWTILE(2);
      } else {
        LAUNCH_AVG_POOL2D_CL_VEC(2);
      }
    } else {
      // No vec8 case: W-tile eligibility above caps max_vs at 4.
      switch (vec_size) {
        case 4:
          LAUNCH_AVG_POOL2D_CL_WTILE(4);
          break;
        case 1:
          LAUNCH_AVG_POOL2D_CL_WTILE(1);
          break;
        default:
          break;
      }
    }
  } else {
    switch (vec_size) {
      case 8:
        LAUNCH_AVG_POOL2D_CL_VEC(8);
        break;
      case 4:
        LAUNCH_AVG_POOL2D_CL_VEC(4);
        break;
      case 2:
        LAUNCH_AVG_POOL2D_CL_VEC(2);
        break;
      default: {
        // Scalar fallback: no vec_size evenly divides channels.
        const int64_t global_range =
            xpuKernelLoopGroupRange(total_elements, group_size) * group_size;
        sycl_kernel_submit<avg_pool2d_channels_last_kernel_impl<
            scalar_t,
            accscalar_t,
            index_t>>(
            global_range,
            group_size,
            queue,
            0,
            top_data,
            bottom_data,
            total_elements,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            divisor_override,
            count_include_pad,
            use_divisor,
            at::detail::IntDivider<unsigned int>(
                static_cast<unsigned int>(channels)),
            at::detail::IntDivider<unsigned int>(
                static_cast<unsigned int>(pooled_width)),
            at::detail::IntDivider<unsigned int>(
                static_cast<unsigned int>(pooled_height)));
        break;
      }
    }
  }
#undef LAUNCH_AVG_POOL2D_CL_VEC
#undef LAUNCH_AVG_POOL2D_CL_WTILE
#undef LAUNCH_AVG_POOL2D_CL_HWTILE
}

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_kernel(
    const int total_elements,
    const Tensor& input,
    const index_t channels,
    const index_t height,
    const index_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& output,
    const int divisor_override,
    const bool count_include_pad,
    const bool use_divisor) {
  scalar_t* top_data = output.mutable_data_ptr<scalar_t>();
  const scalar_t* bottom_data = input.const_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const int64_t group_size = static_cast<int64_t>(
      at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore());
  const int64_t global_range =
      xpuKernelLoopGroupRange(total_elements, group_size) * group_size;

  sycl_kernel_submit<avg_pool2d_kernel_impl<scalar_t, accscalar_t, index_t>>(
      global_range,
      group_size,
      queue,
      0,
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_channels_last_backward_kernel_impl(
    const scalar_t* top_data,
    scalar_t* bottom_data,
    int64_t total_elements,
    int64_t channels,
    int64_t height,
    int64_t width,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP_TYPE(item, index, total_elements, index_t) {
    const int c = index % channels;
    const int w = (index / channels) % width + pad_w;
    const int h = (index / channels / width) % height + pad_h;
    const int n = index / channels / width / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = sycl::min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = sycl::min(w / stride_w + 1, pooled_width);
    accscalar_t gradient = accscalar_t(0);
    const scalar_t* const top_slice =
        top_data + n * channels * pooled_height * pooled_width + c;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend =
            sycl::min(hstart + kernel_h, static_cast<int>(height + pad_h));
        int wend =
            sycl::min(wstart + kernel_w, static_cast<int>(width + pad_w));
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = sycl::max(hstart, 0);
        wstart = sycl::max(wstart, 0);
        hend = sycl::min(hend, static_cast<int>(height));
        wend = sycl::min(wend, static_cast<int>(width));
        if (hstart >= hend || wstart >= wend) {
          continue;
        }
        int divide_factor;
        if (use_divisor) {
          divide_factor = divisor_override;
        } else {
          if (count_include_pad) {
            divide_factor = pool_size;
          } else {
            divide_factor = (hend - hstart) * (wend - wstart);
          }
        }
        gradient +=
            top_slice[(ph * pooled_width + pw) * channels] / divide_factor;
      }
    }
    bottom_data[index] = static_cast<scalar_t>(gradient);
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void avg_pool2d_backward_kernel_impl(
    const scalar_t* top_data,
    scalar_t* bottom_data,
    int64_t total_elements,
    int64_t channels,
    int64_t height,
    int64_t width,
    int pooled_height,
    int pooled_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP_TYPE(item, index, total_elements, index_t) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = sycl::min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = sycl::min(w / stride_w + 1, pooled_width);
    accscalar_t gradient = accscalar_t(0);
    const scalar_t* const top_data_slice =
        top_data + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend =
            sycl::min(hstart + kernel_h, static_cast<int>(height + pad_h));
        int wend =
            sycl::min(wstart + kernel_w, static_cast<int>(width + pad_w));
        int pool_size = (hend - hstart) * (wend - wstart);
        hstart = sycl::max(hstart, 0);
        wstart = sycl::max(wstart, 0);
        hend = sycl::min(hend, static_cast<int>(height));
        wend = sycl::min(wend, static_cast<int>(width));
        if (hstart >= hend || wstart >= wend) {
          continue;
        }
        int divide_factor;
        if (use_divisor) {
          divide_factor = divisor_override;
        } else {
          if (count_include_pad) {
            divide_factor = pool_size;
          } else {
            divide_factor = (hend - hstart) * (wend - wstart);
          }
        }
        gradient += top_data_slice[ph * pooled_width + pw] / divide_factor;
      }
    }
    bottom_data[index] = static_cast<scalar_t>(gradient);
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_backward_channels_last_kernel(
    const index_t total_elements,
    const Tensor& grad_output,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& grad_input,
    const int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  const scalar_t* top_data = grad_output.const_data_ptr<scalar_t>();
  scalar_t* bottom_data = grad_input.mutable_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const int64_t group_size = static_cast<int64_t>(
      at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore());
  const int64_t global_range =
      xpuKernelLoopGroupRange(total_elements, group_size) * group_size;

  sycl_kernel_submit<avg_pool2d_channels_last_backward_kernel_impl<
      scalar_t,
      accscalar_t,
      index_t>>(
      global_range,
      group_size,
      queue,
      0,
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_backward_kernel(
    const index_t total_elements,
    const Tensor& grad_output,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& grad_input,
    const int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  const scalar_t* top_data = grad_output.const_data_ptr<scalar_t>();
  scalar_t* bottom_data = grad_input.mutable_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const int64_t group_size = static_cast<int64_t>(
      at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore());
  const int64_t global_range =
      xpuKernelLoopGroupRange(total_elements, group_size) * group_size;

  sycl_kernel_submit<
      avg_pool2d_backward_kernel_impl<scalar_t, accscalar_t, index_t>>(
      global_range,
      group_size,
      queue,
      0,
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
}

void avg_pool2d_kernel(
    const Tensor& input_,
    int64_t kH_,
    int64_t kW_,
    int64_t dH_,
    int64_t dW_,
    int64_t padH_,
    int64_t padW_,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    const Tensor& output) {
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW_, padW_, dW_, 1, ceil_mode);
  int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH_, padH_, dH_, 1, ceil_mode);
  const auto memory_format = input_.suggest_memory_format();

  Tensor input = input_.contiguous(memory_format);
  const auto count = safe_downcast<int32_t, int64_t>(output.numel());

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value =
      use_divisor ? divisor_override.value() : 0;
  if (count != 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "avg_pool2d_xpu", [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          AT_DISPATCH_INDEX_TYPES(
              at::native::canUse32BitIndexMath(output, INT_MAX)
                  ? ScalarType::Int
                  : ScalarType::Long,
              "avg_pool2d_xpu",
              [&] {
                switch (memory_format) {
                  case MemoryFormat::ChannelsLast: {
                    output.unsafeGetTensorImpl()->empty_tensor_restride(
                        MemoryFormat::ChannelsLast);
                    launch_avg_pool2d_channels_last_kernel<
                        scalar_t,
                        accscalar_t,
                        index_t>(
                        count,
                        input,
                        nInputPlane,
                        inputHeight,
                        inputWidth,
                        outputHeight,
                        outputWidth,
                        kH_,
                        kW_,
                        dH_,
                        dW_,
                        padH_,
                        padW_,
                        output,
                        divisor_override_value,
                        count_include_pad,
                        use_divisor);
                    break;
                  }
                  case MemoryFormat::Contiguous: {
                    launch_avg_pool2d_kernel<scalar_t, accscalar_t, index_t>(
                        count,
                        input,
                        nInputPlane,
                        inputHeight,
                        inputWidth,
                        outputHeight,
                        outputWidth,
                        kH_,
                        kW_,
                        dH_,
                        dW_,
                        padH_,
                        padW_,
                        output,
                        divisor_override_value,
                        count_include_pad,
                        use_divisor);
                    break;
                  }
                  default:
                    TORCH_CHECK(
                        false,
                        "Unsupported memory format. Supports only "
                        "ChannelsLast, Contiguous");
                }
              });
        });
  }
}

void avg_pool2d_backward_kernel(
    const Tensor& gradOutput_,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    const Tensor& gradInput) {
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  if (avg_pool_backward::try_backward(
          gradOutput_,
          input_,
          gradInput,
          kH,
          kW,
          dH,
          dW,
          padH,
          padW,
          ceil_mode,
          count_include_pad,
          divisor_override)) {
    return;
  }

  const auto memory_format = input_.suggest_memory_format();
  const Tensor input = input_.contiguous(memory_format);
  const Tensor gradOutput = gradOutput_.contiguous(memory_format);

  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  const auto count = input.numel();
  if (count == 0) {
    return;
  }
  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value =
      use_divisor ? divisor_override.value() : 0;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "avg_pool2d_backward_xpu", [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;

        AT_DISPATCH_INDEX_TYPES(
            at::native::canUse32BitIndexMath(input, INT_MAX) ? ScalarType::Int
                                                             : ScalarType::Long,
            "avg_pool2d_backward_xpu",
            [&] {
              switch (memory_format) {
                case MemoryFormat::ChannelsLast: {
                  gradInput.unsafeGetTensorImpl()->empty_tensor_restride(
                      MemoryFormat::ChannelsLast);
                  launch_avg_pool2d_backward_channels_last_kernel<
                      scalar_t,
                      accscalar_t,
                      index_t>(
                      count,
                      gradOutput,
                      nInputPlane,
                      inputHeight,
                      inputWidth,
                      outputHeight,
                      outputWidth,
                      kH,
                      kW,
                      dH,
                      dW,
                      padH,
                      padW,
                      gradInput,
                      divisor_override_value,
                      count_include_pad,
                      use_divisor);
                  break;
                }
                case MemoryFormat::Contiguous: {
                  launch_avg_pool2d_backward_kernel<
                      scalar_t,
                      accscalar_t,
                      index_t>(
                      count,
                      gradOutput,
                      nInputPlane,
                      inputHeight,
                      inputWidth,
                      outputHeight,
                      outputWidth,
                      kH,
                      kW,
                      dH,
                      dW,
                      padH,
                      padW,
                      gradInput,
                      divisor_override_value,
                      count_include_pad,
                      use_divisor);
                  break;
                }
                default:
                  TORCH_CHECK(
                      false,
                      "Unsupported memory format. Supports only "
                      "ChannelsLast, Contiguous");
              }
            });
      });
}

} // namespace xpu
} // namespace at::native
