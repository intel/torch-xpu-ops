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

#include <ATen/xpu/XPUContext.h>
#include <comm/Macros.h>
// clang-format off
DISABLE_RETURN_TYPE_WARNING_BEGIN
// clang-format on

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/ops/empty.h>

#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>

#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/DilatedMaxPool2d.h>
#include <ATen/native/xpu/sycl/IntegerDivider.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#ifdef USE_SYCLTLA
#define CUTLASS_SYCL_BUILTIN_ENABLE
// Keep dependency-header diagnostics separate from kernel diagnostics.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#endif
#include <cute/tensor.hpp>

#include <cute/algorithm/reorder.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

namespace at::native::xpu {

namespace max_pool_optimized {
template <typename T, int N>
memory::aligned_vector<T, N> filled(T v) {
  memory::aligned_vector<T, N> out;
#pragma unroll
  for (int j = 0; j < N; ++j)
    out[j] = v;
  return out;
}
template <typename D, typename T, int N>
memory::aligned_vector<D, N> vector_cast(memory::aligned_vector<T, N> v) {
  memory::aligned_vector<D, N> out;
#pragma unroll
  for (int j = 0; j < N; ++j)
    out[j] = D(v[j]);
  return out;
}
template <typename scalar_t = sycl::half>
inline auto unpack_pair(uint32_t word) {
  return vector_cast<float>(
      sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(word));
}
template <typename scalar_t = sycl::half>
inline uint32_t pack_pair(memory::aligned_vector<float, 2> v) {
  return sycl::bit_cast<uint32_t>(vector_cast<scalar_t>(v));
}
inline int compact_index(int64_t value) {
  return value >= 0 && value <= INT_MAX ? int(value) : -1;
}
template <int K, int S, int P>
struct Forward {
  const uint32_t* x;
  uint32_t* y;
  int64_t* idx;
  int H, W, OH, OW, C, count;
  at::detail::IntDivider<uint32_t> dc, dw, dh;
  void operator()(sycl::nd_item<1> it) const {
    for (uint32_t i = it.get_global_linear_id(); i < count;
         i += it.get_global_range(0)) {
      auto pix = dc.div(i), c = i - pix * C, nh = dw.div(pix),
           ox = pix - nh * OW, n = dh.div(nh), oy = nh - n * OH;
      int h0 = int(oy) * S - P, w0 = int(ox) * S - P;
      auto best = filled<float, 2>(-INFINITY);
      auto index = filled<int, 2>(sycl::max(h0, 0) * W + sycl::max(w0, 0));
#pragma unroll
      for (int h = 0; h < K; ++h) {
#pragma unroll
        for (int w = 0; w < K; ++w) {
          int ih = h0 + h, iw = w0 + w;
          if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            auto v = unpack_pair(x[((n * H + ih) * W + iw) * C + c]);
#pragma unroll
            for (int j = 0; j < 2; ++j)
              if (v[j] > best[j] || sycl::isnan(v[j])) {
                best[j] = v[j];
                index[j] = ih * W + iw;
              }
          }
        }
      }
      y[i] = pack_pair(best);
      *reinterpret_cast<memory::aligned_vector<int64_t, 2>*>(
          idx + int64_t(i) * 2) = vector_cast<int64_t>(index);
    }
  }
};

template <int K>
struct LargeForward {
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    if constexpr (K >= 9)
      return sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::grf_size<256>};
    else
      return sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::grf_size<128>};
  }
  const uint32_t* x;
  uint32_t* y;
  int64_t* idx;
  int C;
  sycl::local_accessor<uint32_t, 1> smem;
  void operator()(sycl::nd_item<1> it) const {
    int cb = it.get_group(0) % (C / 16), n = it.get_group(0) / (C / 16),
        ox = it.get_local_id(0) / 16, c = it.get_local_id(0) % 16;
    int row = ox;
    if (row < 12) {
      uint32_t raw[16], pv[16], pi[16];
      auto best = filled<float, 2>(-INFINITY);
      auto bi = filled<int, 2>(0);
#pragma unroll
      for (int xx = 0; xx < 16; ++xx)
        raw[xx] = x[((n * 12 + row) * 16 + xx) * C + cb * 16 + c];
#pragma unroll
      for (int xx = 0; xx < 16; ++xx) {
        auto v = unpack_pair(raw[xx]);
#pragma unroll
        for (int j = 0; j < 2; ++j)
          if (xx % K == 0 || v[j] > best[j] || sycl::isnan(v[j])) {
            best[j] = v[j];
            bi[j] = xx;
          }
        pv[xx] = pack_pair(best);
        pi[xx] = bi[0] | (bi[1] << 8);
        if (xx >= K / 2 && xx < K) {
          int dest = (row * 16 + xx - K / 2) * 16 + c;
          smem[dest] = pv[xx];
          smem[3072 + dest] = pi[xx];
        }
      }
      best = filled<float, 2>(-INFINITY);
      bi = filled<int, 2>(15);
#pragma unroll
      for (int xx = 15; xx >= 0; --xx) {
        auto v = unpack_pair(raw[xx]);
#pragma unroll
        for (int j = 0; j < 2; ++j)
          if (xx == 15 || xx % K == K - 1 ||
              (!sycl::isnan(best[j]) &&
               (v[j] >= best[j] || sycl::isnan(v[j])))) {
            best[j] = v[j];
            bi[j] = xx;
          }
        if (xx > 0 && xx + K / 2 < 16) {
          int hi = sycl::min(15, xx + K - 1);
          auto vv = best;
          auto ii = bi;
          if (xx / K != hi / K) {
            auto bv = unpack_pair(pv[hi]);
            auto bj = memory::aligned_vector<int, 2>{
                {int(pi[hi] & 255), int(pi[hi] >> 8)}};
#pragma unroll
            for (int j = 0; j < 2; ++j)
              if (bv[j] > vv[j] || sycl::isnan(bv[j])) {
                vv[j] = bv[j];
                ii[j] = bj[j];
              }
          }
          int dest = (row * 16 + xx + K / 2) * 16 + c;
          smem[dest] = pack_pair(vv);
          smem[3072 + dest] = ii[0] | (ii[1] << 8);
        }
      }
    }
    it.barrier(sycl::access::fence_space::local_space);
    memory::aligned_vector<float, 2> hv[12];
    memory::aligned_vector<int, 2> hi[12];
#pragma unroll
    for (int h = 0; h < 12; ++h) {
      int a = (h * 16 + ox) * 16 + c;
      hv[h] = unpack_pair(smem[a]);
      uint32_t q = smem[3072 + a];
      hi[h] = memory::aligned_vector<int, 2>{
          {h * 16 + int(q & 255), h * 16 + int(q >> 8)}};
    }
    if constexpr (K == 13) {
      auto best = filled<float, 2>(-INFINITY);
      memory::aligned_vector<int, 2> bi = hi[0];
#pragma unroll
      for (int yy = 0; yy < 12; ++yy) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
          if (hv[yy][j] > best[j] || sycl::isnan(hv[yy][j])) {
            best[j] = hv[yy][j];
            bi[j] = hi[yy][j];
          }
        if (yy >= 6) {
          int dest = ((n * 12 + yy - 6) * 16 + ox) * C + cb * 16 + c;
          y[dest] = pack_pair(best);
          *reinterpret_cast<memory::aligned_vector<int64_t, 2>*>(
              idx + int64_t(dest) * 2) = vector_cast<int64_t>(bi);
        }
      }
      {
        int dest = ((n * 12 + 6) * 16 + ox) * C + cb * 16 + c;
        y[dest] = pack_pair(best);
        *reinterpret_cast<memory::aligned_vector<int64_t, 2>*>(
            idx + int64_t(dest) * 2) = vector_cast<int64_t>(bi);
      }
      best = hv[11];
      bi = hi[11];
#pragma unroll
      for (int yy = 10; yy >= 1; --yy) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
          if (!sycl::isnan(best[j]) &&
              (hv[yy][j] >= best[j] || sycl::isnan(hv[yy][j]))) {
            best[j] = hv[yy][j];
            bi[j] = hi[yy][j];
          }
        if (yy <= 5) {
          int dest = ((n * 12 + yy + 6) * 16 + ox) * C + cb * 16 + c;
          y[dest] = pack_pair(best);
          *reinterpret_cast<memory::aligned_vector<int64_t, 2>*>(
              idx + int64_t(dest) * 2) = vector_cast<int64_t>(bi);
        }
      }
    } else {
#pragma unroll
      for (int h = 0; h < 12; ++h) {
        auto best = filled<float, 2>(-INFINITY);
        memory::aligned_vector<int, 2> index = hi[sycl::max(0, h - K / 2)];
#pragma unroll
        for (int yy = 0; yy < 12; ++yy) {
          if (yy >= h - K / 2 && yy <= h + K / 2) {
#pragma unroll
            for (int j = 0; j < 2; ++j)
              if (hv[yy][j] > best[j] || sycl::isnan(hv[yy][j])) {
                best[j] = hv[yy][j];
                index[j] = hi[yy][j];
              }
          }
        }
        int dest = ((n * 12 + h) * 16 + ox) * C + cb * 16 + c;
        y[dest] = pack_pair(best);
        *reinterpret_cast<memory::aligned_vector<int64_t, 2>*>(idx + dest * 2) =
            vector_cast<int64_t>(index);
      }
    }
  }
};
template <int K>
void large_launch(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& idx) {
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  int groups = x.size(0) * x.size(1) / 32;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> smem(2 * 12 * 16 * 16, h);
    h.parallel_for(
        sycl::nd_range<1>(groups * 256, 256),
        LargeForward<K>{
            (const uint32_t*)x.data_ptr(),
            (uint32_t*)y.data_ptr(),
            idx.data_ptr<int64_t>(),
            int(x.size(1) / 2),
            smem});
  });
}

struct NchwForward {
  const sycl::half* x;
  uint32_t* y;
  int64_t* idx;
  int count;
  void operator()(sycl::nd_item<1> it) const {
    for (int i = it.get_global_linear_id(); i < count;
         i += it.get_global_range(0)) {
      int nc = i / 32, oy = (i / 4) % 8, ox = i % 4 * 2;
      auto best = filled<float, 2>(-INFINITY);
      auto bi = memory::aligned_vector<int, 2>{
          {sycl::max(oy * 2 - 1, 0) * 16 + sycl::max(ox * 2 - 1, 0),
           sycl::max(oy * 2 - 1, 0) * 16 + ox * 2 + 1}};
#pragma unroll
      for (int h = 0; h < 3; ++h) {
        int iy = oy * 2 - 1 + h;
        if (iy >= 0 && iy < 16) {
#pragma unroll
          for (int w = 0; w < 3; ++w) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              int ix = (ox + j) * 2 - 1 + w;
              if (ix >= 0 && ix < 16) {
                float v = float(x[nc * 256 + iy * 16 + ix]);
                if (v > best[j] || sycl::isnan(v)) {
                  best[j] = v;
                  bi[j] = iy * 16 + ix;
                }
              }
            }
          }
        }
      }
      y[i] = pack_pair(best);
      *reinterpret_cast<memory::aligned_vector<int64_t, 2>*>(
          idx + int64_t(i) * 2) = vector_cast<int64_t>(bi);
    }
  }
};

template <typename scalar_t, int K, int P, int CB, int XB>
struct BlockBackward {
  const uint32_t* go;
  const int64_t* idx;
  uint32_t* gi;
  int H, W, OH, OW, C, CS, XT, YT, CT;
  at::detail::IntDivider<uint32_t> dct, dxt, dyt;
  sycl::local_accessor<int, 1> li;
  void operator()(sycl::nd_item<1> it) const {
    uint32_t g = it.get_group(0), tmp = dct.div(g);
    int cb = (g - tmp * CT) * CB;
    g = tmp;
    tmp = dxt.div(g);
    int xb = (g - tmp * XT) * XB;
    g = tmp;
    tmp = dyt.div(g);
    int yb = g - tmp * YT, n = tmp;
    constexpr int NH = K == 2 ? 1 : 2, NW = K == 2 ? XB : XB + 1,
                  DELTA = (K == 3 && P == 0) ? -1 : 0;
    for (int j = it.get_local_id(0); j < XB * CB; j += 256) {
      int c = j / XB, xx = j % XB, pw = xb + DELTA + xx;
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        int ph = yb + DELTA + h;
        li[(h * NW + xx) * (CB + 1) + c] =
            (cb + c < C * 2 && ph >= 0 && ph < OH && pw >= 0 && pw < OW)
            ? compact_index(idx[((n * C * 2 + cb + c) * OH + ph) * OW + pw])
            : -1;
      }
    }
    if constexpr (K == 3) {
      int c = it.get_local_id(0);
      if (c < CB) {
        int pw = xb + DELTA + XB;
#pragma unroll
        for (int h = 0; h < NH; ++h) {
          int ph = yb + DELTA + h;
          li[(h * NW + XB) * (CB + 1) + c] =
              (cb + c < C * 2 && ph >= 0 && ph < OH && pw >= 0 && pw < OW)
              ? compact_index(idx[((n * C * 2 + cb + c) * OH + ph) * OW + pw])
              : -1;
        }
      }
    }
    it.barrier(sycl::access::fence_space::local_space);
    for (int j = it.get_local_id(0); j < XB * CB / 2; j += 256) {
      int c = j % (CB / 2) * 2, xx = j / (CB / 2), pwbase = xb + xx,
          iy = yb * 2, ix = pwbase * 2;
      if (cb + c < C * 2 && ix < W) {
        memory::aligned_vector<float, 2> acc[4] = {
            filled<float, 2>(0),
            filled<float, 2>(0),
            filled<float, 2>(0),
            filled<float, 2>(0)};
#pragma unroll
        for (int hh = 0; hh < NH; ++hh) {
#pragma unroll
          for (int ww = 0; ww < NH; ++ww) {
            int ph = yb + DELTA + hh, pw = pwbase + DELTA + ww;
            if (ph >= 0 && ph < OH && pw >= 0 && pw < OW) {
              auto val = unpack_pair<scalar_t>(
                  go[((n * OH + ph) * OW + pw) * CS + (cb + c) / 2]);
              int a = (hh * NW + xx + ww) * (CB + 1) + c;
              int i0 = li[a], i1 = li[a + 1];
#pragma unroll
              for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
                for (int dx = 0; dx < 2; ++dx) {
                  if (K == 2 ||
                      ((P == 1 ? (hh == 0 || dy == 1) : (hh == 1 || dy == 0)) &&
                       (P == 1 ? (ww == 0 || dx == 1)
                               : (ww == 1 || dx == 0)))) {
                    int target = (iy + dy) * W + ix + dx;
                    if (i0 == target)
                      acc[dy * 2 + dx][0] += val[0];
                    if (i1 == target)
                      acc[dy * 2 + dx][1] += val[1];
                  }
                }
              }
            }
          }
        }
#pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
          for (int dx = 0; dx < 2; ++dx)
            if (iy + dy < H && ix + dx < W)
              gi[((n * H + iy + dy) * W + ix + dx) * C + (cb + c) / 2] =
                  pack_pair<scalar_t>(acc[dy * 2 + dx]);
        }
      }
    }
  }
};
template <typename scalar_t, int K, int P, int CB, int XB>
void block_launch_impl(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  int H = x.size(2), W = x.size(3), C = x.size(1) / 2, OH = v.size(2),
      OW = v.size(3), XT = (W + 2 * XB - 1) / (2 * XB), YT = (H + 1) / 2,
      CT = (C * 2 + CB - 1) / CB;
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int, 1> li((K == 2 ? XB : 2 * (XB + 1)) * (CB + 1), h);
    h.parallel_for(
        sycl::nd_range<1>(x.size(0) * XT * YT * CT * 256, 256),
        BlockBackward<scalar_t, K, P, CB, XB>{
            (const uint32_t*)v.data_ptr(),
            idx.data_ptr<int64_t>(),
            (uint32_t*)out.data_ptr(),
            H,
            W,
            OH,
            OW,
            C,
            int(v.stride(3) / 2),
            XT,
            YT,
            CT,
            at::detail::IntDivider<uint32_t>(CT),
            at::detail::IntDivider<uint32_t>(XT),
            at::detail::IntDivider<uint32_t>(YT),
            li});
  });
}
template <typename scalar_t, int K, int P>
void block_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  if (x.size(3) <= 16)
    block_launch_impl<scalar_t, K, P, 64, 8>(x, v, idx, out);
  else if (x.size(3) <= 32)
    block_launch_impl<scalar_t, K, P, 64, 16>(x, v, idx, out);
  else
    block_launch_impl<scalar_t, K, P, 64, 32>(x, v, idx, out);
}

template <int K, int S, int P>
void forward_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx) {
  int H = x.size(2), W = x.size(3), C = x.size(1) / 2, OH = v.size(2),
      OW = v.size(3), count = v.numel() / 2;
  int groups = std::min((count + 255) / 256, 1280);
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>(groups * 256, 256),
      Forward<K, S, P>{
          (const uint32_t*)x.data_ptr(),
          (uint32_t*)v.data_ptr(),
          idx.data_ptr<int64_t>(),
          H,
          W,
          OH,
          OW,
          C,
          count,
          at::detail::IntDivider<uint32_t>(C),
          at::detail::IntDivider<uint32_t>(OW),
          at::detail::IntDivider<uint32_t>(OH)});
}
template <int K, int S, int P>
struct ForwardTile {
  const uint32_t* x;
  uint32_t* y;
  int64_t* idx;
  int H, W, OH, OW, C, CT, XT, YT, count, NP;
  at::detail::IntDivider<uint32_t> dc, dx, dy;
  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const int id =
        it.get_group_linear_id() * 8 + it.get_sub_group().get_group_linear_id();
    if (id >= count)
      return;
    const int lane = it.get_local_linear_id() % 16;
    const int q = dc.div(id), cb = (id - q * CT) * 32, c = cb + lane * 2;
    const int r = dx.div(q), ox = (q - r * XT) * (S == 1 ? 2 : 3);
    const int n = dy.div(r), oy = (r - n * YT) * (S == 1 ? 2 : 1);
    constexpr int NY = S == 1 ? 2 : 1, NX = S == 1 ? 2 : 3;
    memory::aligned_vector<sycl::half, 2> best[NY * NX];
    memory::aligned_vector<int, 2> bi[NY * NX];
#pragma unroll
    for (int yy = 0; yy < NY; ++yy) {
#pragma unroll
      for (int xx = 0; xx < NX; ++xx) {
        best[yy * NX + xx] = filled<sycl::half, 2>(sycl::half(-INFINITY));
        bi[yy * NX + xx] = filled<int, 2>(
            sycl::max((oy + yy) * S - P, 0) * W +
            sycl::max((ox + xx) * S - P, 0));
      }
    }
#pragma unroll
    for (int h = 0; h < K + NY - 1; ++h) {
      const int ih = oy * S - P + h;
      if (ih >= 0 && ih < H) {
        constexpr int Columns = S == 1 ? 4 : 8;
        memory::aligned_vector<uint32_t, Columns> raw;
#ifdef USE_SYCLTLA
        if (C % 8 == 0) {
          using Copy = std::conditional_t<
              S == 1,
              cute::XE_2D_U32x4x16_LD_N,
              cute::XE_2D_U32x8x16_LD_N>;
          Copy::copy(
              x + ((n * H + ih) * W) * (C / 2),
              C * 2,
              W,
              C * 2,
              cute::intel::coord_t{cb / 2, ox * S - P},
              raw.val);
        } else if (S == 2 && NP % 4 == 0) {
#pragma unroll
          for (int parity = 0; parity < 4; ++parity) {
            const int pos = (n * H + ih) * W + ox * S - P + parity;
            memory::aligned_vector<uint32_t, 2> v;
            cute::XE_2D_U32x2x16_LD_N::copy(
                x,
                C * 8,
                NP / 4,
                C * 8,
                cute::intel::coord_t{(pos & 3) * (C / 2) + cb / 2, pos >> 2},
                v.val);
            raw[parity] = v[0];
            if constexpr (Columns == 8)
              raw[parity + 4] = v[1];
          }
        } else
#endif
        {
#pragma unroll
          for (int w = 0; w < Columns; ++w) {
            const int iw = ox * S - P + w;
            raw[w] = c < C && iw >= 0 && iw < W
                ? x[((n * H + ih) * W + iw) * (C / 2) + c / 2]
                : 0;
          }
        }
#pragma unroll
        for (int w = 0; w < K + (NX - 1) * S; ++w) {
          const int iw = ox * S - P + w;
          if (iw >= 0 && iw < W) {
            auto v =
                sycl::bit_cast<memory::aligned_vector<sycl::half, 2>>(raw[w]);
#pragma unroll
            for (int yy = 0; yy < NY; ++yy) {
#pragma unroll
              for (int xx = 0; xx < NX; ++xx) {
                if (h >= yy && h < yy + K && w >= xx * S && w < xx * S + K) {
#pragma unroll
                  for (int j = 0; j < 2; ++j)
                    if (v[j] > best[yy * NX + xx][j] ||
                        sycl::isnan(float(v[j]))) {
                      best[yy * NX + xx][j] = v[j];
                      bi[yy * NX + xx][j] = ih * W + iw;
                    }
                }
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int yy = 0; yy < NY; ++yy) {
#pragma unroll
      for (int xx = 0; xx < NX; ++xx)
        if (c < C && oy + yy < OH && ox + xx < OW) {
          const int pos = ((n * OH + oy + yy) * OW + ox + xx) * (C / 2) + c / 2;
          y[pos] = sycl::bit_cast<uint32_t>(best[yy * NX + xx]);
          *reinterpret_cast<memory::aligned_vector<int64_t, 2>*>(
              idx + int64_t(pos) * 2) = vector_cast<int64_t>(bi[yy * NX + xx]);
        }
    }
  }
};
template <int K, int S, int P>
void forward_tile_launch(const Tensor& x, const Tensor& v, const Tensor& idx) {
  const int C = x.size(1), CT = (C + 31) / 32,
            XT = (v.size(3) + (S == 1 ? 1 : 2)) / (S == 1 ? 2 : 3);
  const int YT = (v.size(2) + (S == 1 ? 1 : 0)) / (S == 1 ? 2 : 1),
            count = x.size(0) * CT * XT * YT;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 7) / 8 * 128, 128),
      ForwardTile<K, S, P>{
          reinterpret_cast<const uint32_t*>(x.data_ptr()),
          reinterpret_cast<uint32_t*>(v.data_ptr()),
          idx.data_ptr<int64_t>(),
          int(x.size(2)),
          int(x.size(3)),
          int(v.size(2)),
          int(v.size(3)),
          C,
          CT,
          XT,
          YT,
          count,
          int(x.size(0) * x.size(2) * x.size(3)),
          at::detail::IntDivider<uint32_t>(CT),
          at::detail::IntDivider<uint32_t>(XT),
          at::detail::IntDivider<uint32_t>(YT)});
}
#ifdef USE_SYCLTLA
template <typename scalar_t>
struct BackwardNonoverlap {
  const uint64_t* idx;
  const uint32_t* go;
  uint32_t* gi;
  int H, W, OH, OW, C, CS, CT, XT, count;
  at::detail::IntDivider<uint32_t> dc, dx, dy;
  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const int id = it.get_group_linear_id();
    const int q = dc.div(id), cb = (id - q * CT) * 32;
    const int r = dx.div(q), px = (q - r * XT) * 4;
    const int n = dy.div(r), ph = r - n * OH, HW = OH * OW,
              pbase = ph * OW + px;
    uint32_t indices[4][2];
#pragma unroll
    for (int parity = 0; parity < 2; ++parity) {
      memory::aligned_vector<uint32_t, 8> raw;
      cute::XE_2D_U32x16x8_LD_T::copy(
          idx + int64_t(n) * C * HW,
          HW * 16,
          C / 2,
          HW * 16,
          cute::intel::coord_t{(pbase + parity * HW) * 2, cb / 2},
          raw.val);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        indices[j][parity] = raw[2 * j + 1] == 0 ? raw[2 * j] : UINT_MAX;
    }
    memory::aligned_vector<uint32_t, 4> values;
    cute::XE_2D_U32x4x16_LD_N::copy(
        go + int64_t(n) * HW * (CS / 2),
        C * 2,
        HW,
        CS * 2,
        cute::intel::coord_t{cb / 2, pbase},
        values.val);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      auto v = sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(values[j]);
#pragma unroll
      for (int c = 0; c < 2; ++c)
        v[c] = scalar_t(float(v[c]) + 0.f);
      values[j] = sycl::bit_cast<uint32_t>(v);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        memory::aligned_vector<uint32_t, 4> packed;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int target = (ph * 2 + h) * W + (px + j) * 2 + w;
          const uint32_t mask = (indices[j][0] == target ? 0xffffu : 0u) |
              (indices[j][1] == target ? 0xffff0000u : 0u);
          packed[j] = values[j] & mask;
        }
        cute::XE_2D_U32x4x16_ST_N::copy(
            gi + ((n * H + ph * 2 + h) * W + w) * (C / 2),
            C * 2,
            OW,
            C * 4,
            cute::intel::coord_t{cb / 2, px},
            packed.val);
      }
    }
  }
};
template <typename scalar_t>
void nonoverlap_launch(
    const Tensor& x,
    const Tensor& v,
    const Tensor& idx,
    const Tensor& out) {
  const int C = x.size(1), CT = (C + 31) / 32, XT = (v.size(3) + 3) / 4,
            count = x.size(0) * v.size(2) * CT * XT;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>(count * 16, 16),
      BackwardNonoverlap<scalar_t>{
          reinterpret_cast<const uint64_t*>(idx.data_ptr()),
          reinterpret_cast<const uint32_t*>(v.data_ptr()),
          reinterpret_cast<uint32_t*>(out.data_ptr()),
          int(x.size(2)),
          int(x.size(3)),
          int(v.size(2)),
          int(v.size(3)),
          C,
          int(v.stride(3)),
          CT,
          XT,
          count,
          at::detail::IntDivider<uint32_t>(CT),
          at::detail::IntDivider<uint32_t>(XT),
          at::detail::IntDivider<uint32_t>(v.size(2))});
}
template <typename scalar_t>
struct WholeRowBackward {
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<16>,
        sycl::ext::intel::experimental::grf_size<128>};
  }
  const uint64_t* idx;
  const uint32_t* go;
  uint32_t* gi;
  int H, OH, C, CS;
  void operator()(sycl::nd_item<3> it) const {
    const int n = it.get_group(0), yb = it.get_group(1);
    const int cb =
        (it.get_group(2) * 8 + it.get_sub_group().get_group_linear_id()) * 32;
    if (cb >= C)
      return;
    const int HW = OH * 8;
    using IndexCopy = cute::XE_LOAD_2D_TRANSPOSE<32, 32, 8>;
    auto source = cute::make_tensor(
        cute::make_gmem_ptr(
            reinterpret_cast<const uint32_t*>(idx + int64_t(n) * C * HW)),
        cute::make_layout(
            cute::make_shape(HW * 2, C),
            cute::make_stride(cute::_1{}, HW * 2)));
    cute::Copy_Traits<
        IndexCopy,
        cute::_0,
        cute::_1,
        uint32_t,
        cute::Stride<cute::_0, cute::_0>>
        index_copy(source);
    float overlap[9][2];
    scalar_t partial[3][9][2];
#pragma unroll
    for (int pp = 0; pp < 9; ++pp) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        overlap[pp][c] = 0.f;
#pragma unroll
        for (int k = 0; k < 3; ++k)
          partial[k][pp][c] = scalar_t(0);
      }
    }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int ph = yb - 1 + hh;
      if (ph >= 0 && ph < OH) {
        uint16_t indices[10][2];
        uint32_t values[10];
        values[0] = 0;
        values[9] = 0;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          indices[0][c] = 65535;
          indices[9][c] = 65535;
        }
#pragma unroll
        for (int block = 0; block < 2; ++block) {
          memory::aligned_vector<uint32_t, 16> raw, paired;
          index_copy.template update_payload<32>(
              cute::make_coord((ph * 8 + block * 4) * 2, cb));
          IndexCopy::copy(index_copy.payload, raw.val);
          auto input = cute::make_tensor(
              cute::make_rmem_ptr(raw.val), cute::make_layout(cute::Int<16>{}));
          auto output = cute::make_tensor(
              cute::make_rmem_ptr(paired.val),
              cute::make_layout(cute::Int<16>{}));
          using TVShape =
              cute::Shape<cute::_16, cute::Shape<cute::_2, cute::_8>>;
          cute::reorder(
              input,
              output,
              cute::Layout<
                  TVShape,
                  cute::Stride<cute::_1, cute::Stride<cute::_16, cute::_32>>>{},
              cute::Layout<
                  TVShape,
                  cute::Stride<cute::_2, cute::Stride<cute::_1, cute::_32>>>{});
#pragma unroll
          for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int c = 0; c < 2; ++c)
              indices[block * 4 + j + 1][c] =
                  paired[j * 4 + c + 2] == 0 && paired[j * 4 + c] < 65535
                  ? uint16_t(paired[j * 4 + c])
                  : uint16_t(65535);
          }
        }
        memory::aligned_vector<uint32_t, 8> raw_values;
        cute::XE_2D_U32x8x16_LD_N::copy(
            go + int64_t(n) * HW * (CS / 2),
            C * 2,
            HW,
            CS * 2,
            cute::intel::coord_t{cb / 2, ph * 8},
            raw_values.val);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          values[j + 1] = raw_values[j];
#pragma unroll
        for (int ww = 0; ww < 2; ++ww) {
#pragma unroll
          for (int pp = 0; pp < 9; ++pp) {
            auto v = sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
                values[pp + ww]);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
              for (int w = 0; w < 2; ++w) {
                if ((hh == 1 || h == 0) && (ww == 1 || w == 0)) {
                  const int target = (yb * 2 + h) * 17 + pp * 2 + w;
#pragma unroll
                  for (int c = 0; c < 2; ++c) {
                    const scalar_t add =
                        indices[pp + ww][c] == target ? v[c] : scalar_t(0);
                    if (h * 2 + w == 0)
                      overlap[pp][c] += float(add);
                    else
                      partial[h * 2 + w - 1][pp][c] += add;
                  }
                }
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (yb * 2 + h < H) {
#pragma unroll
        for (int w = 0; w < 2; ++w) {
          memory::aligned_vector<uint32_t, 8> packed;
#pragma unroll
          for (int pp = 0; pp < 8; ++pp) {
            memory::aligned_vector<scalar_t, 2> v;
#pragma unroll
            for (int c = 0; c < 2; ++c)
              v[c] = h * 2 + w == 0 ? scalar_t(overlap[pp][c])
                                    : partial[h * 2 + w - 1][pp][c];
            packed[pp] = sycl::bit_cast<uint32_t>(v);
          }
          auto dst = gi + ((n * H + yb * 2 + h) * 17 + w) * (C / 2);
          cute::XE_2D_U32x8x16_ST_N::copy(
              dst,
              C * 2,
              (18 - w) / 2,
              C * 4,
              cute::intel::coord_t{cb / 2, 0},
              packed.val);
          if (w == 0) {
            memory::aligned_vector<scalar_t, 2> v;
#pragma unroll
            for (int c = 0; c < 2; ++c)
              v[c] = h == 0 ? scalar_t(overlap[8][c]) : partial[1][8][c];
            uint32_t tail = sycl::bit_cast<uint32_t>(v);
            cute::XE_2D_U32x1x16_ST_N::copy(
                dst, C * 2, 9, C * 4, cute::intel::coord_t{cb / 2, 8}, &tail);
          }
        }
      }
    }
  }
};
template <typename scalar_t>
void whole_row_launch(
    const Tensor& x,
    const Tensor& v,
    const Tensor& idx,
    const Tensor& out) {
  const int C = x.size(1), YT = (x.size(2) + 1) / 2;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(x.size(0), YT, ((C / 32 + 7) / 8) * 128),
          sycl::range<3>(1, 1, 128)),
      WholeRowBackward<scalar_t>{
          reinterpret_cast<const uint64_t*>(idx.data_ptr()),
          reinterpret_cast<const uint32_t*>(v.data_ptr()),
          reinterpret_cast<uint32_t*>(out.data_ptr()),
          int(x.size(2)),
          int(v.size(2)),
          C,
          int(v.stride(3))});
}
template <
    typename scalar_t,
    int K,
    int P,
    bool Aligned = false,
    bool Packed = false>
struct BackwardTile {
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::sub_group_size<16>,
        sycl::ext::intel::experimental::grf_size<128>};
  }
  const uint64_t* idx;
  const uint32_t* go;
  uint32_t* gi;
  int H, W, OH, OW, C, CS, CT, XT, NP;
  at::detail::IntDivider<uint32_t> dc;
  void operator()(sycl::nd_item<3> it) const {
    const int id =
        it.get_group(2) * 8 + it.get_sub_group().get_group_linear_id();
    if (id >= CT * XT)
      return;
    const int lane = it.get_local_linear_id() % 16;
    const int q = dc.div(id), cb = (id - q * CT) * 32, px = q * 7;
    const int n = it.get_group(0), yb = it.get_group(1);
    const int HW = OH * OW;
    constexpr int D = K == 3 && P == 0 ? -1 : 0;
    using IndexCopy = cute::XE_LOAD_2D_TRANSPOSE<32, Packed ? 32 : 16, 8>;
    auto index_source = cute::make_tensor(
        cute::make_gmem_ptr(
            reinterpret_cast<const uint32_t*>(idx + n * C * HW)),
        cute::make_layout(
            cute::make_shape(HW * (Packed ? 2 : 4), C / (Packed ? 1 : 2)),
            cute::make_stride(cute::_1{}, HW * (Packed ? 2 : 4))));
    cute::Copy_Traits<
        IndexCopy,
        cute::_0,
        cute::_1,
        uint32_t,
        cute::Stride<cute::_0, cute::_0>>
        index_copy(index_source);
    // Explicit initialization avoids copying zero arrays from constant memory.
    float overlap[7][2];
    scalar_t partial[4][7][2];
#pragma unroll
    for (int pp = 0; pp < 7; ++pp) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        overlap[pp][c] = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k)
          partial[k][pp][c] = scalar_t(0);
      }
    }
#pragma unroll
    for (int hh = 0; hh < (K == 2 ? 1 : 2); ++hh) {
      const int ph = yb + D + hh;
      if (ph >= 0 && ph < OH) {
        const int pw = px + D, pbase = ph * OW + pw;
        uint16_t indices[8][2];
#pragma unroll
        for (int block = 0; block < 2; ++block) {
          if constexpr (Packed) {
            memory::aligned_vector<uint32_t, 16> raw, paired;
            index_copy.template update_payload<32>(
                cute::make_coord((pbase + block * 4) * 2, cb));
            IndexCopy::copy(index_copy.payload, raw.val);
            auto input = cute::make_tensor(
                cute::make_rmem_ptr(raw.val),
                cute::make_layout(cute::Int<16>{}));
            auto output = cute::make_tensor(
                cute::make_rmem_ptr(paired.val),
                cute::make_layout(cute::Int<16>{}));
            using TVShape =
                cute::Shape<cute::_16, cute::Shape<cute::_2, cute::_8>>;
            cute::reorder(
                input,
                output,
                cute::Layout<
                    TVShape,
                    cute::
                        Stride<cute::_1, cute::Stride<cute::_16, cute::_32>>>{},
                cute::Layout<
                    TVShape,
                    cute::
                        Stride<cute::_2, cute::Stride<cute::_1, cute::_32>>>{});
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
              for (int c = 0; c < 2; ++c)
                indices[block * 4 + j][c] =
                    paired[j * 4 + c + 2] == 0 && paired[j * 4 + c] < 65535
                    ? uint16_t(paired[j * 4 + c])
                    : uint16_t(65535);
            }
          } else {
#pragma unroll
            for (int parity = 0; parity < 2; ++parity) {
              memory::aligned_vector<uint32_t, 8> raw;
              index_copy.template update_payload<32>(cute::make_coord(
                  (pbase + block * 4 + parity * HW) * 2, cb / 2));
              IndexCopy::copy(index_copy.payload, raw.val);
#pragma unroll
              for (int j = 0; j < 4; ++j)
                indices[block * 4 + j][parity] =
                    raw[j * 2 + 1] == 0 && raw[j * 2] < 65535
                    ? uint16_t(raw[j * 2])
                    : uint16_t(65535);
            }
          }
        }
        memory::aligned_vector<uint32_t, 8> values;
        if constexpr (Aligned) {
          cute::XE_2D_U32x8x16_LD_N::copy(
              go + n * HW * (CS / 2),
              C * 2,
              HW,
              CS * 2,
              cute::intel::coord_t{cb / 2, pbase},
              values.val);
        } else if (NP % 4 == 0) {
#pragma unroll
          for (int parity = 0; parity < 4; ++parity) {
            const int pos = n * HW + pbase + parity;
            memory::aligned_vector<uint32_t, 2> raw;
            cute::XE_2D_U32x2x16_LD_N::copy(
                go,
                (3 * CS + C) * 2,
                NP / 4,
                CS * 8,
                cute::intel::coord_t{(pos & 3) * (CS / 2) + cb / 2, pos >> 2},
                raw.val);
            values[parity] = raw[0];
            values[parity + 4] = raw[1];
          }
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j)
            values[j] = cb + lane * 2 < C && pw + j >= 0 && pw + j < OW
                ? go[((n * OH + ph) * OW + pw + j) * (CS / 2) + cb / 2 + lane]
                : 0;
        }
#pragma unroll
        for (int ww = 0; ww < (K == 2 ? 1 : 2); ++ww) {
#pragma unroll
          for (int pp = 0; pp < 7; ++pp) {
            auto v = sycl::bit_cast<memory::aligned_vector<scalar_t, 2>>(
                values[pp + ww]);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
              for (int w = 0; w < 2; ++w) {
                if (K == 2 ||
                    ((P == 1 ? (hh == 0 || h == 1) : (hh == 1 || h == 0)) &&
                     (P == 1 ? (ww == 0 || w == 1) : (ww == 1 || w == 0)))) {
                  const uint16_t target = (yb * 2 + h) * W + (px + pp) * 2 + w;
#pragma unroll
                  for (int c = 0; c < 2; ++c) {
                    const scalar_t add = pw + pp + ww >= 0 &&
                            pw + pp + ww < OW && indices[pp + ww][c] == target
                        ? v[c]
                        : scalar_t(0);
                    if (K == 3 && h * 2 + w == (P == 1 ? 3 : 0))
                      overlap[pp][c] += float(add);
                    else
                      partial[h * 2 + w][pp][c] += add;
                  }
                }
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (yb * 2 + h < H) {
        if constexpr (Packed || !Aligned) {
          uint32_t row[14];
#pragma unroll
          for (int pp = 0; pp < 7; ++pp) {
#pragma unroll
            for (int w = 0; w < 2; ++w) {
              memory::aligned_vector<scalar_t, 2> v;
#pragma unroll
              for (int c = 0; c < 2; ++c)
                v[c] = K == 3 && h * 2 + w == (P == 1 ? 3 : 0)
                    ? scalar_t(overlap[pp][c])
                    : partial[h * 2 + w][pp][c];
              row[pp * 2 + w] = sycl::bit_cast<uint32_t>(v);
            }
          }
          if constexpr (Packed) {
#pragma unroll
            for (int block = 0; block < 2; ++block) {
              memory::aligned_vector<uint32_t, 8> packed;
#pragma unroll
              for (int j = 0; j < 8; ++j)
                packed[j] = block * 8 + j < 14 ? row[block * 8 + j] : 0;
              cute::XE_2D_U32x8x16_ST_N::copy(
                  gi + ((n * H + yb * 2 + h) * W) * 12,
                  48,
                  sycl::min(W, px * 2 + 14),
                  48,
                  cute::intel::coord_t{0, px * 2 + block * 8},
                  packed.val);
            }
          } else {
            // Four spatial positions form a row with a 16-byte aligned pitch.
            const int first = (n * H + yb * 2 + h) * W + px * 2;
            const int valid = sycl::min(14, W - px * 2);
#pragma unroll
            for (int parity = 0; parity < 4; ++parity) {
              if (parity < valid) {
                memory::aligned_vector<uint32_t, 4> packed;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                  packed[j] = j * 4 + parity < 14 ? row[j * 4 + parity] : 0;
                const int pos = first + parity;
                cute::XE_2D_U32x4x16_ST_N::copy(
                    gi,
                    ((pos & 3) + 1) * C * 2,
                    (pos + (valid - 1 - parity) / 4 * 4) / 4 + 1,
                    C * 8,
                    cute::intel::coord_t{(pos & 3) * (C / 2) + cb / 2, pos / 4},
                    packed.val);
              }
            }
          }
        } else {
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            memory::aligned_vector<uint32_t, 8> packed;
            packed[7] = 0;
#pragma unroll
            for (int pp = 0; pp < 7; ++pp) {
              memory::aligned_vector<scalar_t, 2> v;
#pragma unroll
              for (int c = 0; c < 2; ++c)
                v[c] = K == 3 && h * 2 + w == (P == 1 ? 3 : 0)
                    ? scalar_t(overlap[pp][c])
                    : partial[h * 2 + w][pp][c];
              packed[pp] = sycl::bit_cast<uint32_t>(v);
            }
            if constexpr (Aligned) {
              cute::XE_2D_U32x8x16_ST_N::copy(
                  gi + ((n * H + yb * 2 + h) * W + w) * (C / 2),
                  C * 2,
                  sycl::min((W + 1 - w) / 2, px + 7),
                  C * 4,
                  cute::intel::coord_t{cb / 2, px},
                  packed.val);
            } else {
#pragma unroll
              for (int pp = 0; pp < 7; ++pp)
                if (cb + lane * 2 < C && (px + pp) * 2 + w < W)
                  gi[((n * H + yb * 2 + h) * W + (px + pp) * 2 + w) * (C / 2) +
                     cb / 2 + lane] = packed[pp];
            }
          }
        }
      }
    }
  }
};
template <
    typename scalar_t,
    int K,
    int P,
    bool Aligned = false,
    bool Packed = false>
void backward_tile_impl(
    const Tensor& x,
    const Tensor& v,
    const Tensor& idx,
    const Tensor& out) {
  const int C = x.size(1), CT = (C + 31) / 32, XT = (x.size(3) + 13) / 14,
            YT = (x.size(2) + 1) / 2;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(x.size(0), YT, ((CT * XT + 7) / 8) * 128),
          sycl::range<3>(1, 1, 128)),
      BackwardTile<scalar_t, K, P, Aligned, Packed>{
          reinterpret_cast<const uint64_t*>(idx.data_ptr()),
          reinterpret_cast<const uint32_t*>(v.data_ptr()),
          reinterpret_cast<uint32_t*>(out.data_ptr()),
          int(x.size(2)),
          int(x.size(3)),
          int(v.size(2)),
          int(v.size(3)),
          C,
          int(v.stride(3)),
          CT,
          XT,
          int(x.size(0) * v.size(2) * v.size(3)),
          at::detail::IntDivider<uint32_t>(CT)});
}
template <typename scalar_t, int K, int P>
void backward_tile_launch(
    const Tensor& x,
    const Tensor& v,
    const Tensor& idx,
    const Tensor& out) {
  if constexpr (K == 3) {
    if (x.size(1) == 24 && x.size(3) % 4 == 0 && v.stride(3) % 8 == 0 &&
        v.size(2) * v.size(3) % 2 == 0) {
      backward_tile_impl<scalar_t, K, P, true, true>(x, v, idx, out);
      return;
    }
  }
  if (x.size(1) % 8 == 0 && v.stride(3) % 8 == 0)
    backward_tile_impl<scalar_t, K, P, true>(x, v, idx, out);
  else
    backward_tile_impl<scalar_t, K, P, false>(x, v, idx, out);
}

#endif
#ifdef USE_SYCLTLA
template <int C, int H>
struct CompactIndex {
  const uint64_t* src;
  uint16_t* dst;
  int count;
  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const int id =
        it.get_group_linear_id() * 8 + it.get_sub_group().get_group_linear_id();
    if (id >= count)
      return;
    constexpr int HW = H * H, CT = (C + 31) / 32, CP = (C + 63) / 64 * 64,
                  Tiles = (HW + 3) / 4;
    const int cb = id % CT * 32, t = id / CT, n = t / Tiles, p0 = t % Tiles * 4;
    memory::aligned_vector<uint8_t, 2> codes[4];
#pragma unroll
    for (int parity = 0; parity < 2; ++parity) {
      memory::aligned_vector<uint32_t, 8> raw;
      cute::XE_2D_U32x16x8_LD_T::copy(
          src + int64_t(n) * C * HW,
          HW * 16,
          C / 2,
          HW * 16,
          cute::intel::coord_t{(p0 + parity * HW) * 2, cb / 2},
          raw.val);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t delta = raw[j * 2] - (p0 + j) + H + 1;
        codes[j][parity] = raw[j * 2 + 1] == 0 && delta <= 2 * H + 2
            ? uint8_t(delta)
            : uint8_t(255);
      }
    }
    memory::aligned_vector<uint16_t, 4> packed;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      packed[j] = sycl::bit_cast<uint16_t>(codes[j]);
    cute::XE_2D_U16x4x16_ST_N::copy(
        dst + n * HW * (CP / 2),
        CP,
        HW,
        CP,
        cute::intel::coord_t{cb / 2, p0},
        packed.val);
  }
};
template <typename scalar_t, int C, int H>
struct CompactBackward {
  const memory::aligned_vector<scalar_t, 8>* go;
  const memory::aligned_vector<uint8_t, 8>* idx;
  memory::aligned_vector<scalar_t, 8>* gi;
  int CS, count;
  [[sycl::reqd_sub_group_size(C == 216 ? 32 : 16)]] void operator()(
      sycl::nd_item<1> it) const {
    constexpr int T = (H + 1) / 2, CP = (C + 63) / 64 * 64;
    for (int i = it.get_global_linear_id(); i < count;
         i += it.get_global_range(0)) {
      const int c = i % (C / 8), tile = i / (C / 8), x = tile % T * 2,
                y = tile / T % T * 2, n = tile / (T * T);
      memory::aligned_vector<float, 8> acc[4]{};
#pragma unroll
      for (int h = 0; h < 4; ++h) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const int py = y + h - 1, px = x + w - 1;
          if (py >= 0 && py < H && px >= 0 && px < H) {
            const int pix = (n * H + py) * H + px;
            const auto ids = idx[pix * (CP / 8) + c];
            const auto val = go[pix * CS + c];
#pragma unroll
            for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
              for (int dx = 0; dx < 2; ++dx) {
                if (h >= dy && h <= dy + 2 && w >= dx && w <= dx + 2) {
#pragma unroll
                  for (int j = 0; j < 8; ++j)
                    if (ids[j] == (dy + 2 - h) * H + dx + 2 - w)
                      acc[dy * 2 + dx][j] += float(val[j]);
                }
              }
            }
          }
        }
      }
#pragma unroll
      for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
        for (int dx = 0; dx < 2; ++dx)
          if (y + dy < H && x + dx < H)
            gi[((n * H + y + dy) * H + x + dx) * (C / 8) + c] =
                vector_cast<scalar_t>(acc[dy * 2 + dx]);
      }
    }
  }
};
template <typename scalar_t, int C, int H>
void compact_backward_launch(
    const Tensor& x,
    const Tensor& v,
    const Tensor& idx,
    const Tensor& out) {
  constexpr int CP = (C + 63) / 64 * 64;
  const int count = x.size(0) * ((H * H + 3) / 4) * ((C + 31) / 32);
  auto tmp = at::empty({x.size(0) * H * H * CP}, v.options().dtype(at::kByte));
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(
      sycl::nd_range<1>((count + 7) / 8 * 128, 128),
      CompactIndex<C, H>{
          reinterpret_cast<const uint64_t*>(idx.data_ptr()),
          reinterpret_cast<uint16_t*>(tmp.data_ptr()),
          count});
  const int total = x.size(0) * ((H + 1) / 2) * ((H + 1) / 2) * (C / 8);
  q.parallel_for(
      sycl::nd_range<1>(std::min((total + 255) / 256, 1280) * 256, 256),
      CompactBackward<scalar_t, C, H>{
          reinterpret_cast<const memory::aligned_vector<scalar_t, 8>*>(
              v.data_ptr()),
          reinterpret_cast<const memory::aligned_vector<uint8_t, 8>*>(
              tmp.data_ptr()),
          reinterpret_cast<memory::aligned_vector<scalar_t, 8>*>(
              out.data_ptr()),
          int(v.stride(3) / 8),
          total});
}
#endif
inline bool bmg_device() {
  const auto arch =
      c10::xpu::getCurrentXPUStream()
          .queue()
          .get_device()
          .get_info<
              sycl::ext::oneapi::experimental::info::device::architecture>();
  using A = sycl::ext::oneapi::experimental::architecture;
  return arch == A::intel_gpu_bmg_g21 || arch == A::intel_gpu_bmg_g31;
}
inline bool parameters(
    at::IntArrayRef k,
    at::IntArrayRef s,
    at::IntArrayRef p,
    at::IntArrayRef d,
    int& K,
    int& S,
    int& P) {
  if (k.empty() || p.empty() || d.empty() || k.size() > 2 || s.size() > 2 ||
      p.size() > 2 || d.size() > 2)
    return false;
  if ((k.size() == 2 && k[0] != k[1]) || (s.size() == 2 && s[0] != s[1]) ||
      (p.size() == 2 && p[0] != p[1]) || d[0] != 1 ||
      (d.size() == 2 && d[1] != 1))
    return false;
  if (k[0] > 13 || k[0] < 1 || p[0] < 0 || p[0] > 6 ||
      (!s.empty() && (s[0] < 1 || s[0] > 2)))
    return false;
  K = k[0];
  S = s.empty() ? K : s[0];
  P = p[0];
  return (S == 2 &&
          ((K == 1 && P == 0) || (K == 2 && P == 0) ||
           (K == 3 && (P == 0 || P == 1)))) ||
      (S == 1 && (K == 3 || K == 5 || K == 9 || K == 13) && P == K / 2);
}
inline bool forward(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    at::IntArrayRef k,
    at::IntArrayRef s,
    at::IntArrayRef p,
    at::IntArrayRef d) {
  int K, S, P;
  if (x.dim() != 4 || x.scalar_type() != at::kHalf ||
      v.scalar_type() != at::kHalf || idx.scalar_type() != at::kLong ||
      !x.numel() || !v.numel() || x.numel() > INT_MAX || v.numel() > INT_MAX ||
      x.size(1) % 2 || !parameters(k, s, p, d, K, S, P))
    return false;
  if (reinterpret_cast<uintptr_t>(x.data_ptr()) % 64 ||
      reinterpret_cast<uintptr_t>(v.data_ptr()) % 4 ||
      reinterpret_cast<uintptr_t>(idx.data_ptr()) % 16)
    return false;
  bool nhwc = x.is_contiguous(at::MemoryFormat::ChannelsLast) &&
      v.is_contiguous(at::MemoryFormat::ChannelsLast) &&
      idx.is_contiguous(at::MemoryFormat::ChannelsLast);
  bool nchw = x.is_contiguous() && v.is_contiguous() && idx.is_contiguous() &&
      x.size(2) == 16 && x.size(3) == 16 && v.size(2) == 8 && v.size(3) == 8 &&
      K == 3 && S == 2 && P == 1;
  if (!nhwc && !nchw)
    return false;
  if (K >= 5 && (!nhwc || x.size(2) != 12 || x.size(3) != 16 || x.size(1) % 32))
    return false;
  if (S == 1 && K == 3 &&
      (x.size(2) * x.size(3) >= 65535 || v.sizes() != x.sizes()))
    return false;
  if (nchw) {
    c10::xpu::getCurrentXPUStream().queue().parallel_for(
        sycl::nd_range<1>(1280 * 256, 256),
        NchwForward{
            (const sycl::half*)x.data_ptr(),
            (uint32_t*)v.data_ptr(),
            idx.data_ptr<int64_t>(),
            int(v.numel() / 2)});
    return true;
  }
  if (K >= 5) {
    if (K == 5)
      large_launch<5>(x, v, idx);
    else if (K == 9)
      large_launch<9>(x, v, idx);
    else
      large_launch<13>(x, v, idx);
  } else if (S == 1)
    forward_tile_launch<3, 1, 1>(x, v, idx);
  else if (K == 1)
    forward_launch<1, 2, 0>(x, v, idx);
  else if (K == 2)
    forward_tile_launch<2, 2, 0>(x, v, idx);
  else if (P == 0)
    forward_tile_launch<3, 2, 0>(x, v, idx);
  else
    forward_tile_launch<3, 2, 1>(x, v, idx);
  return true;
}
template <typename scalar_t>
bool dispatch_backward(
    const Tensor& x,
    const Tensor& v,
    const Tensor& idx,
    const Tensor& out,
    int K,
    int S,
    int P) {
  const int64_t C = x.size(1), H = x.size(2), W = x.size(3), OW = v.size(3),
                CS = v.stride(3);
  if (H * W >= 65535) {
    if (S == 1)
      return false;
    if (K == 2)
      block_launch<scalar_t, 2, 0>(x, v, idx, out);
    else if (P == 0)
      block_launch<scalar_t, 3, 0>(x, v, idx, out);
    else
      block_launch<scalar_t, 3, 1>(x, v, idx, out);
    return true;
  }
#ifdef USE_SYCLTLA
  if (K == 2 && C % 32 == 0 && CS % 8 == 0 && H % 2 == 0 && W % 2 == 0) {
    nonoverlap_launch<scalar_t>(x, v, idx, out);
    return true;
  }
  if (K == 3 && S == 2 && P == 0 && W == 17 && OW == 8 && C >= 128 &&
      C % 32 == 0 && CS % 8 == 0 && out.numel() >= (10LL << 20)) {
    whole_row_launch<scalar_t>(x, v, idx, out);
    return true;
  }
  if (S == 1) {
    if (C == 864)
      compact_backward_launch<scalar_t, 864, 11>(x, v, idx, out);
    else if (C == 432)
      compact_backward_launch<scalar_t, 432, 21>(x, v, idx, out);
    else
      compact_backward_launch<scalar_t, 216, 42>(x, v, idx, out);
  } else if (K == 2)
    backward_tile_launch<scalar_t, 2, 0>(x, v, idx, out);
  else if (P == 0)
    backward_tile_launch<scalar_t, 3, 0>(x, v, idx, out);
  else
    backward_tile_launch<scalar_t, 3, 1>(x, v, idx, out);
#else
  if (S == 1)
    return false;
  if (K == 2)
    block_launch<scalar_t, 2, 0>(x, v, idx, out);
  else if (P == 0)
    block_launch<scalar_t, 3, 0>(x, v, idx, out);
  else
    block_launch<scalar_t, 3, 1>(x, v, idx, out);
#endif
  return true;
}
inline bool backward(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out,
    at::IntArrayRef k,
    at::IntArrayRef s,
    at::IntArrayRef p,
    at::IntArrayRef d) {
  int K, S, P;
  if (x.dim() != 4 || v.dim() != 4 || idx.dim() != 4 ||
      (x.scalar_type() != at::kHalf && x.scalar_type() != at::kBFloat16) ||
      v.scalar_type() != x.scalar_type() ||
      out.scalar_type() != x.scalar_type() || idx.scalar_type() != at::kLong ||
      !x.numel() || !v.numel() || x.numel() > INT_MAX || v.numel() > INT_MAX ||
      !parameters(k, s, p, d, K, S, P) || K > 3 || K < 2)
    return false;
  int64_t C = x.size(1), H = x.size(2), W = x.size(3), OH = v.size(2),
          OW = v.size(3), CS = v.stride(3);
  if (OH * OW > (1 << 20) || CS > (1 << 23) ||
      ((C % 8 || CS % 8) && C >= (1 << 21)))
    return false;
  if (C % 2 || !x.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      !out.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      !idx.is_contiguous() || idx.sizes() != v.sizes() ||
      out.sizes() != x.sizes() || v.stride(1) != 1 || CS < C || CS % 2 ||
      v.stride(2) != OW * CS || v.stride(0) != OH * OW * CS ||
      (x.size(0) * OH * OW - 1) * CS + C > INT_MAX)
    return false;
  // The 2D block messages require an aligned base; unusual offsets use the
  // generic kernel.
  if (reinterpret_cast<uintptr_t>(v.data_ptr()) % 64 ||
      reinterpret_cast<uintptr_t>(idx.data_ptr()) % 64 ||
      reinterpret_cast<uintptr_t>(out.data_ptr()) % 64)
    return false;
  if (S == 1 &&
      !(H == W && OH == H && OW == W && CS % 8 == 0 &&
        ((C == 216 && H == 42) || (C == 432 && H == 21) ||
         (C == 864 && H == 11))))
    return false;
  if (x.scalar_type() == at::kHalf)
    return dispatch_backward<sycl::half>(x, v, idx, out, K, S, P);
  return dispatch_backward<c10::BFloat16>(x, v, idx, out, K, S, P);
}
} // namespace max_pool_optimized

#ifdef __SYCL_DEVICE_ONLY__
static inline int p_start(
    int size,
    int pad,
    int kernel,
    int dilation,
    int stride) {
  return (size + pad < ((kernel - 1) * dilation + 1))
      ? 0
      : (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static inline int p_end(int size, int pad, int pooled_size, int stride) {
  return std::min((size + pad) / stride + 1, pooled_size);
}
#endif

static inline bool can_use_int32_nhwc(
    int64_t nbatch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t in_stride_n,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w) {
  constexpr int64_t int_max = std::numeric_limits<int>::max();

  int64_t max_intra_batch = (height ? (height - 1) * in_stride_h : 0) +
      (width ? (width - 1) * in_stride_w : 0) +
      (channels ? (channels - 1) * in_stride_c : 0);

  int64_t max_input_offset =
      (nbatch ? (nbatch - 1) * in_stride_n : 0) + max_intra_batch;

  if (max_input_offset > int_max)
    return false;

  int64_t out_batch_stride = pooled_height * pooled_width * channels;
  if ((nbatch ? (nbatch - 1) * out_batch_stride : 0) > int_max)
    return false;

  if (height * width > int_max)
    return false;

  return true;
}

static inline bool can_use_int32_nchw(
    int64_t nbatch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width) {
  int64_t hw = height * width;
  return can_use_int32_nhwc(
      nbatch,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      channels * hw, // in_stride_n
      hw, // in_stride_c
      width, // in_stride_h
      1 // in_stride_w
  );
}

template <typename scalar_t, bool is_channels_last, typename index_t = int>
struct MaxPool2dKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);

    do {
      if (desc.glb_problem < cfg_.problem_) {
        index_t outputIndex = desc.glb_problem;
        index_t batch = outputIndex / stride_;
        index_t plane, outputH, outputW;
        index_t load_offset, store_offset;
        if constexpr (is_channels_last) {
          plane = outputIndex % numPlane_;
          outputH = outputIndex / numPlane_ / outputSizeW_ % outputSizeH_;
          outputW = outputIndex / numPlane_ % outputSizeW_;
          store_offset = batch * outputSizeH_ * outputSizeW_ * numPlane_ +
              plane + outputH * outputSizeW_ * numPlane_ + outputW * numPlane_;
        } else {
          plane = (outputIndex / outputSizeH_ / outputSizeW_) % numPlane_;
          outputH = outputIndex / outputSizeW_ % outputSizeH_;
          outputW = outputIndex % outputSizeW_;
          store_offset = batch * numPlane_ * outputSizeH_ * outputSizeW_ +
              plane * outputSizeH_ * outputSizeW_ + outputH * outputSizeW_ +
              outputW;
        }
        scalar_t maxVal = at::numeric_limits<scalar_t>::lower_bound();
        index_t StartH = outputH * dH_ - padH_;
        index_t StartW = outputW * dW_ - padW_;
        index_t EndH =
            std::min(StartH + (kH_ - 1) * dilationH_ + 1, inputSizeH_);
        index_t EndW =
            std::min(StartW + (kW_ - 1) * dilationW_ + 1, inputSizeW_);
        while (StartH < 0)
          StartH += dilationH_;
        while (StartW < 0)
          StartW += dilationW_;
        index_t maxIndex = StartH * inputSizeW_ + StartW;
#pragma unroll
        for (index_t h = StartH; h < EndH; h += dilationH_) {
#pragma unroll
          for (index_t w = StartW; w < EndW; w += dilationW_) {
            if constexpr (is_channels_last) {
              load_offset = batch * inputSizeH_ * inputSizeW_ * numPlane_ +
                  plane + h * inputSizeW_ * numPlane_ + w * numPlane_;
            } else {
              load_offset = batch * numPlane_ * inputSizeH_ * inputSizeW_ +
                  plane * inputSizeH_ * inputSizeW_ + h * inputSizeW_ + w;
            }
            scalar_t val = input_[load_offset];
            if ((static_cast<scalar_t>(val) > maxVal) || at::_isnan(val)) {
              maxIndex = h * inputSizeW_ + w;
              maxVal = static_cast<scalar_t>(val);
            }
          }
        }
        indices_[store_offset] = maxIndex;
        output_[store_offset] = static_cast<scalar_t>(maxVal);
      }
    } while (cfg_.next(item, desc));
  }
  MaxPool2dKernelFunctor(
      scalar_t* output,
      int64_t* indices,
      const scalar_t* input,
      index_t numPlane,
      index_t inputSizeH,
      index_t inputSizeW,
      index_t outputSizeH,
      index_t outputSizeW,
      int kH,
      int kW,
      int dH,
      int dW,
      int padH,
      int padW,
      int dilationH,
      int dilationW,
      index_t stride,
      BatchKernelConfig cfg)
      : output_(output),
        indices_(indices),
        input_(input),
        numPlane_(numPlane),
        inputSizeH_(inputSizeH),
        inputSizeW_(inputSizeW),
        outputSizeH_(outputSizeH),
        outputSizeW_(outputSizeW),
        kH_(kH),
        kW_(kW),
        dH_(dH),
        dW_(dW),
        padH_(padH),
        padW_(padW),
        dilationH_(dilationH),
        dilationW_(dilationW),
        stride_(stride),
        cfg_(cfg) {}

 private:
  scalar_t* output_;
  int64_t* indices_;
  const scalar_t* input_;
  index_t numPlane_;
  index_t inputSizeH_;
  index_t inputSizeW_;
  index_t outputSizeH_;
  index_t outputSizeW_;
  int kH_;
  int kW_;
  int dH_;
  int dW_;
  int padH_;
  int padW_;
  int dilationH_;
  int dilationW_;
  index_t stride_;
  BatchKernelConfig cfg_;
};

template <
    typename scalar_t,
    typename vec_t,
    int vec_size,
    typename index_t = int>
struct MaxPool2dChannelLastVec {
  void operator()(sycl::nd_item<1> item) const {
    for (auto outputIndex = item.get_global_linear_id();
         outputIndex < numBatch_ * stride_ / vec_size;
         outputIndex += item.get_local_range(0) * item.get_group_range(0)) {
      index_t batch = outputIndex / (stride_ / vec_size);
      index_t plane, outputH, outputW;
      int64_t load_offset, store_offset;
      plane = outputIndex % (numPlane_ / vec_size);
      outputH =
          outputIndex / (numPlane_ / vec_size) / outputSizeW_ % outputSizeH_;
      outputW = outputIndex / (numPlane_ / vec_size) % outputSizeW_;
      store_offset = outputIndex;

      vec_t maxVal_vec;
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        maxVal_vec[i] = at::numeric_limits<scalar_t>::lower_bound();
      }
      index_t StartH = outputH * dH_ - padH_;
      index_t StartW = outputW * dW_ - padW_;
      index_t EndH = std::min(StartH + (kH_ - 1) * dilationH_ + 1, inputSizeH_);
      index_t EndW = std::min(StartW + (kW_ - 1) * dilationW_ + 1, inputSizeW_);
      while (StartH < 0)
        StartH += dilationH_;
      while (StartW < 0)
        StartW += dilationW_;
      int64_t maxIndex[vec_size];
      for (int i = 0; i < vec_size; i++) {
        maxIndex[i] = StartH * inputSizeW_ + StartW;
      }
      for (index_t h = StartH; h < EndH; h += dilationH_) {
        for (index_t w = StartW; w < EndW; w += dilationW_) {
          load_offset =
              batch * inputSizeH_ * inputSizeW_ * numPlane_ / vec_size + plane +
              h * inputSizeW_ * numPlane_ / vec_size + w * numPlane_ / vec_size;
          vec_t val_vec = input_vec_[load_offset];
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            if ((static_cast<scalar_t>(val_vec[i]) > maxVal_vec[i]) ||
                at::_isnan(val_vec[i])) {
              maxIndex[i] = h * inputSizeW_ + w;
              maxVal_vec[i] = static_cast<scalar_t>(val_vec[i]);
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        indices_[store_offset * vec_size + i] = maxIndex[i];
      }
      output_vec_[store_offset] = maxVal_vec;
    }
  }
  MaxPool2dChannelLastVec(
      vec_t* output_vec,
      int64_t* indices,
      const vec_t* input_vec,
      index_t numBatch,
      index_t numPlane,
      index_t inputSizeH,
      index_t inputSizeW,
      index_t outputSizeH,
      index_t outputSizeW,
      int kH,
      int kW,
      int dH,
      int dW,
      int padH,
      int padW,
      int dilationH,
      int dilationW,
      index_t stride)
      : output_vec_(output_vec),
        indices_(indices),
        input_vec_(input_vec),
        numBatch_(numBatch),
        numPlane_(numPlane),
        inputSizeH_(inputSizeH),
        inputSizeW_(inputSizeW),
        outputSizeH_(outputSizeH),
        outputSizeW_(outputSizeW),
        kH_(kH),
        kW_(kW),
        dH_(dH),
        dW_(dW),
        padH_(padH),
        padW_(padW),
        dilationH_(dilationH),
        dilationW_(dilationW),
        stride_(stride) {}

 private:
  vec_t* output_vec_;
  int64_t* indices_;
  const vec_t* input_vec_;
  index_t numBatch_;
  index_t numPlane_;
  index_t inputSizeH_;
  index_t inputSizeW_;
  index_t outputSizeH_;
  index_t outputSizeW_;
  int kH_;
  int kW_;
  int dH_;
  int dW_;
  int padH_;
  int padW_;
  int dilationH_;
  int dilationW_;
  index_t stride_;
};

template <typename scalar_t, bool is_channels_last, typename index_t = int>
struct MaxPool2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);

    do {
      if (desc.glb_problem < cfg_.problem_) {
        index_t batch = desc.glb_problem / out_n_stride_;
        index_t outputIndex = desc.glb_problem;
        if constexpr (is_channels_last) {
          index_t plane = outputIndex % numPlane_;
          int64_t index = indices_[outputIndex];
          index_t gI_offset = batch * in_n_stride_ + plane + index * numPlane_;
          atomicAdd(
              (sycl_global_ptr<scalar_t>)&gradInput_[gI_offset],
              gradOutput_[outputIndex]);
        } else {
          index_t plane = outputIndex / out_cf_c_stride_ % numPlane_;
          int64_t index = indices_[outputIndex];
          index_t gI_offset =
              batch * in_n_stride_ + plane * in_cf_c_stride_ + index;
          atomicAdd(
              (sycl_global_ptr<scalar_t>)&gradInput_[gI_offset],
              gradOutput_[outputIndex]);
        }
      }
    } while (cfg_.next(item, desc));
  }
  MaxPool2dBackwardKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* gradOutput,
      const int64_t* indices,
      index_t numPlane,
      index_t gradInputSizeH,
      index_t gradInputSizeW,
      index_t gradOutputSizeH,
      index_t gradOutputSizeW,
      int64_t gradOutputSize,
      index_t out_cf_c_stride,
      index_t in_cf_c_stride,
      index_t out_n_stride,
      index_t in_n_stride,
      BatchKernelConfig cfg)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        indices_(indices),
        numPlane_(numPlane),
        gradInputSizeH_(gradInputSizeH),
        gradInputSizeW_(gradInputSizeW),
        gradOutputSizeH_(gradOutputSizeH),
        gradOutputSizeW_(gradOutputSizeW),
        gradOutputSize_(gradOutputSize),
        out_cf_c_stride_(out_cf_c_stride),
        in_cf_c_stride_(in_cf_c_stride),
        out_n_stride_(out_n_stride),
        in_n_stride_(in_n_stride),
        cfg_(cfg) {}

 private:
  scalar_t* gradInput_;
  const scalar_t* gradOutput_;
  const int64_t* indices_;
  index_t numPlane_;
  index_t gradInputSizeH_;
  index_t gradInputSizeW_;
  index_t gradOutputSizeH_;
  index_t gradOutputSizeW_;
  int64_t gradOutputSize_;
  index_t out_cf_c_stride_;
  index_t in_cf_c_stride_;
  index_t out_n_stride_;
  index_t in_n_stride_;
  BatchKernelConfig cfg_;
};

template <
    typename scalar_t,
    typename accscalar_t,
    bool is_channels_last,
    typename index_t = int>
struct MaxPool2dBackwardDeterministicKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);
    do {
      if (desc.glb_problem < cfg_.problem_) {
        index_t inputIndex = desc.glb_problem;
        index_t batch = inputIndex / in_n_stride_;
        index_t plane;
        index_t input_hw_index;
        if constexpr (is_channels_last) {
          plane = inputIndex % numPlane_;
          input_hw_index = ((inputIndex % in_n_stride_) - plane) / numPlane_;
        } else {
          plane = inputIndex / in_cf_c_stride_ % numPlane_;
          input_hw_index = ((inputIndex % in_n_stride_)) % in_cf_c_stride_;
        }
        index_t inputW = input_hw_index % gradInputSizeW_;
        index_t inputH = input_hw_index / gradInputSizeW_;
        int phstart =
            p_start(inputH, pad_h_, kernel_h_, dilation_h_, stride_h_);
        int phend = p_end(inputH, pad_h_, gradOutputSizeH_, stride_h_);
        int pwstart =
            p_start(inputW, pad_w_, kernel_w_, dilation_w_, stride_w_);
        int pwend = p_end(inputW, pad_w_, gradOutputSizeW_, stride_w_);
        accscalar_t grad = accscalar_t(0);
        if constexpr (is_channels_last) {
          index_t offset = batch * out_n_stride_ + plane;
          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              if (indices_[offset + (ph * gradOutputSizeW_ + pw) * numPlane_] ==
                  input_hw_index) {
                grad += static_cast<accscalar_t>(
                    gradOutput_
                        [offset + (ph * gradOutputSizeW_ + pw) * numPlane_]);
              }
            }
          }
        } else {
          index_t offset = batch * out_n_stride_ + plane * out_cf_c_stride_;
          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              if (indices_[offset + ph * gradOutputSizeW_ + pw] ==
                  input_hw_index) {
                grad += static_cast<accscalar_t>(
                    gradOutput_[offset + ph * gradOutputSizeW_ + pw]);
              }
            }
          }
        }
        gradInput_[inputIndex] = static_cast<scalar_t>(grad);
      }
    } while (cfg_.next(item, desc));
  }
  MaxPool2dBackwardDeterministicKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* gradOutput,
      const int64_t* indices,
      index_t numPlane,
      index_t gradInputSizeH,
      index_t gradInputSizeW,
      index_t gradOutputSizeH,
      index_t gradOutputSizeW,
      int64_t gradInputSize,
      index_t out_cf_c_stride,
      index_t in_cf_c_stride,
      index_t out_n_stride,
      index_t in_n_stride,
      int kernel_h,
      int kernel_w,
      int stride_h,
      int stride_w,
      int pad_h,
      int pad_w,
      int dilation_h,
      int dilation_w,
      BatchKernelConfig cfg)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        indices_(indices),
        numPlane_(numPlane),
        gradInputSizeH_(gradInputSizeH),
        gradInputSizeW_(gradInputSizeW),
        gradOutputSizeH_(gradOutputSizeH),
        gradOutputSizeW_(gradOutputSizeW),
        gradInputSize_(gradInputSize),
        out_cf_c_stride_(out_cf_c_stride),
        in_cf_c_stride_(in_cf_c_stride),
        out_n_stride_(out_n_stride),
        in_n_stride_(in_n_stride),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_h_(pad_h),
        pad_w_(pad_w),
        dilation_h_(dilation_h),
        dilation_w_(dilation_w),
        cfg_(cfg) {}

 private:
  scalar_t* gradInput_;
  const scalar_t* gradOutput_;
  const int64_t* indices_;
  index_t numPlane_;
  index_t gradInputSizeH_;
  index_t gradInputSizeW_;
  index_t gradOutputSizeH_;
  index_t gradOutputSizeW_;
  int64_t gradInputSize_;
  index_t out_cf_c_stride_;
  index_t in_cf_c_stride_;
  index_t out_n_stride_;
  index_t in_n_stride_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int dilation_h_;
  int dilation_w_;
  BatchKernelConfig cfg_;
};

template <
    typename scalar_t,
    typename accscalar_t,
    typename vec_t,
    int vec_size,
    typename index_t = int>
struct MaxPool2dBackwardChannelLastVec {
  void operator()(sycl::nd_item<1> item) const {
    for (auto inputIndex = item.get_global_linear_id();
         inputIndex < gradInputSize_ / vec_size;
         inputIndex += item.get_local_range(0) * item.get_group_range(0)) {
      index_t batch = inputIndex / (in_n_stride_ / vec_size);
      index_t plane;
      int64_t input_hw_index;

      plane = inputIndex % (numPlane_ / vec_size);

      input_hw_index = ((inputIndex % (in_n_stride_ / vec_size)) - plane) /
          (numPlane_ / vec_size);

      index_t inputW = input_hw_index % gradInputSizeW_;
      index_t inputH = input_hw_index / gradInputSizeW_;
      int phstart = p_start(inputH, pad_h_, kernel_h_, dilation_h_, stride_h_);
      int phend = p_end(inputH, pad_h_, gradOutputSizeH_, stride_h_);
      int pwstart = p_start(inputW, pad_w_, kernel_w_, dilation_w_, stride_w_);
      int pwend = p_end(inputW, pad_w_, gradOutputSizeW_, stride_w_);
      accscalar_t grad_acc[vec_size];
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        grad_acc[i] = accscalar_t(0);
      }

      index_t offset = batch * out_n_stride_ / vec_size + plane;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          index_t load_offset = offset +
              ph * gradOutputSizeW_ * numPlane_ / vec_size +
              pw * numPlane_ / vec_size;
          vec_t gout_val_vec = gradOutput_[load_offset];
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            if (indices_[load_offset * vec_size + i] == input_hw_index) {
              grad_acc[i] += static_cast<accscalar_t>(gout_val_vec[i]);
            }
          }
        }
      }

      vec_t grad_vec;
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        grad_vec[i] = static_cast<scalar_t>(grad_acc[i]);
      }
      gradInput_[inputIndex] = grad_vec;
    }
  }
  MaxPool2dBackwardChannelLastVec(
      vec_t* gradInput,
      const vec_t* gradOutput,
      const int64_t* indices,
      index_t numPlane,
      index_t gradInputSizeH,
      index_t gradInputSizeW,
      index_t gradOutputSizeH,
      index_t gradOutputSizeW,
      int64_t gradInputSize,
      index_t out_n_stride,
      index_t in_n_stride,
      int kernel_h,
      int kernel_w,
      int stride_h,
      int stride_w,
      int pad_h,
      int pad_w,
      int dilation_h,
      int dilation_w)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        indices_(indices),
        numPlane_(numPlane),
        gradInputSizeH_(gradInputSizeH),
        gradInputSizeW_(gradInputSizeW),
        gradOutputSizeH_(gradOutputSizeH),
        gradOutputSizeW_(gradOutputSizeW),
        gradInputSize_(gradInputSize),
        out_n_stride_(out_n_stride),
        in_n_stride_(in_n_stride),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_h_(pad_h),
        pad_w_(pad_w),
        dilation_h_(dilation_h),
        dilation_w_(dilation_w) {}

 private:
  vec_t* gradInput_;
  const vec_t* gradOutput_;
  const int64_t* indices_;
  index_t numPlane_;
  index_t gradInputSizeH_;
  index_t gradInputSizeW_;
  index_t gradOutputSizeH_;
  index_t gradOutputSizeW_;
  int64_t gradInputSize_;
  index_t out_n_stride_;
  index_t in_n_stride_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int dilation_h_;
  int dilation_w_;
};

#define LAUNCH_MAXPOOL_CHANNEL_LAST_VEC(                                    \
    scalar_t,                                                               \
    vec_size,                                                               \
    index_t,                                                                \
    num_wg,                                                                 \
    wg_size,                                                                \
    queue,                                                                  \
    output,                                                                 \
    indices,                                                                \
    input,                                                                  \
    numBatch,                                                               \
    numPlane,                                                               \
    inputSizeH,                                                             \
    inputSizeW,                                                             \
    outputSizeH,                                                            \
    outputSizeW,                                                            \
    kH,                                                                     \
    kW,                                                                     \
    dH,                                                                     \
    dW,                                                                     \
    padH,                                                                   \
    padW,                                                                   \
    dilationH,                                                              \
    dilationW,                                                              \
    stride)                                                                 \
  {                                                                         \
    using vec_t = memory::aligned_vector<scalar_t, vec_size>;               \
    vec_t* output_vec = reinterpret_cast<vec_t*>(output);                   \
    const vec_t* input_vec = reinterpret_cast<const vec_t*>(input);         \
    auto kfn = MaxPool2dChannelLastVec<scalar_t, vec_t, vec_size, index_t>( \
        output_vec,                                                         \
        indices,                                                            \
        input_vec,                                                          \
        numBatch,                                                           \
        numPlane,                                                           \
        inputSizeH,                                                         \
        inputSizeW,                                                         \
        outputSizeH,                                                        \
        outputSizeW,                                                        \
        kH,                                                                 \
        kW,                                                                 \
        dH,                                                                 \
        dW,                                                                 \
        padH,                                                               \
        padW,                                                               \
        dilationH,                                                          \
        dilationW,                                                          \
        stride);                                                            \
    sycl_kernel_submit(num_wg* wg_size, wg_size, queue, kfn);               \
  }

template <typename scalar_t, bool is_channels_last, typename index_t = int>
void launch_max_pool2d_kernel(
    scalar_t* output,
    int64_t* indices,
    const scalar_t* input,
    index_t numBatch,
    index_t numPlane,
    index_t inputSizeH,
    index_t inputSizeW,
    index_t outputSizeH,
    index_t outputSizeW,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int64_t outputSize = static_cast<int64_t>(numBatch) *
      static_cast<int64_t>(numPlane) * static_cast<int64_t>(outputSizeH) *
      static_cast<int64_t>(outputSizeW);
  int64_t stride = static_cast<int64_t>(numPlane) *
      static_cast<int64_t>(outputSizeH) * static_cast<int64_t>(outputSizeW);
  int vec_size = 1;
  int thread_slots = at::xpu::getDeviceHWThreads();
  int64_t num_sub_wg;
  int64_t wg_size = at::xpu::getDeviceMaxWorkGroupSize();
  int64_t num_wg;
  if constexpr (is_channels_last) {
    for (vec_size =
             std::min(8, memory::can_vectorize_up_to<scalar_t>((char*)input));
         vec_size > 1;
         vec_size /= 2) {
      if (numPlane % vec_size != 0) {
        continue;
      }
      num_sub_wg = outputSize / vec_size / at::xpu::getDeviceMaxSubGroupSize();
      if (2 * num_sub_wg > thread_slots) {
        int total_thread = outputSize / vec_size;
        num_wg = (total_thread + wg_size - 1) / wg_size;
        break;
      }
    }
    switch (vec_size) {
      case 8:
        LAUNCH_MAXPOOL_CHANNEL_LAST_VEC(
            scalar_t,
            8,
            index_t,
            num_wg,
            wg_size,
            queue,
            output,
            indices,
            input,
            numBatch,
            numPlane,
            inputSizeH,
            inputSizeW,
            outputSizeH,
            outputSizeW,
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            dilationH,
            dilationW,
            stride);
        return;
      case 4:
        LAUNCH_MAXPOOL_CHANNEL_LAST_VEC(
            scalar_t,
            4,
            index_t,
            num_wg,
            wg_size,
            queue,
            output,
            indices,
            input,
            numBatch,
            numPlane,
            inputSizeH,
            inputSizeW,
            outputSizeH,
            outputSizeW,
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            dilationH,
            dilationW,
            stride);
        return;
      case 2:
        LAUNCH_MAXPOOL_CHANNEL_LAST_VEC(
            scalar_t,
            2,
            index_t,
            num_wg,
            wg_size,
            queue,
            output,
            indices,
            input,
            numBatch,
            numPlane,
            inputSizeH,
            inputSizeW,
            outputSizeH,
            outputSizeW,
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            dilationH,
            dilationW,
            stride);
        return;
      default:
        break;
    };
  }
  using KernelClass =
      MaxPool2dKernelFunctor<scalar_t, is_channels_last, index_t>;

  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      1, outputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive);
  auto kfn = KernelClass(
      output,
      indices,
      input,
      numPlane,
      inputSizeH,
      inputSizeW,
      outputSizeH,
      outputSizeW,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      stride,
      cfg);
  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
}

#define LAUNCH_MAXPOOL_BACKWARD_CHANNEL_LAST_VEC(                              \
    scalar_t,                                                                  \
    vec_size,                                                                  \
    index_t,                                                                   \
    num_wg,                                                                    \
    wg_size,                                                                   \
    queue,                                                                     \
    gradInput,                                                                 \
    gradOutput,                                                                \
    indices,                                                                   \
    numPlane,                                                                  \
    gradInputSizeH,                                                            \
    gradInputSizeW,                                                            \
    gradOutputSizeH,                                                           \
    gradOutputSizeW,                                                           \
    gradInputSize,                                                             \
    out_n_stride,                                                              \
    in_n_stride,                                                               \
    kernel_h,                                                                  \
    kernel_w,                                                                  \
    stride_h,                                                                  \
    stride_w,                                                                  \
    pad_h,                                                                     \
    pad_w,                                                                     \
    dilation_h,                                                                \
    dilation_w)                                                                \
  {                                                                            \
    using accscalar_t = at::acc_type_device<scalar_t, kXPU>;                   \
    using vec_t = memory::aligned_vector<scalar_t, vec_size>;                  \
    const vec_t* grad_output_vec = reinterpret_cast<const vec_t*>(gradOutput); \
    vec_t* grad_input_vec = reinterpret_cast<vec_t*>(gradInput);               \
    auto kfn = MaxPool2dBackwardChannelLastVec<                                \
        scalar_t,                                                              \
        accscalar_t,                                                           \
        vec_t,                                                                 \
        vec_size,                                                              \
        index_t>(                                                              \
        grad_input_vec,                                                        \
        grad_output_vec,                                                       \
        indices,                                                               \
        numPlane,                                                              \
        gradInputSizeH,                                                        \
        gradInputSizeW,                                                        \
        gradOutputSizeH,                                                       \
        gradOutputSizeW,                                                       \
        gradInputSize,                                                         \
        out_n_stride,                                                          \
        in_n_stride,                                                           \
        kernel_h,                                                              \
        kernel_w,                                                              \
        stride_h,                                                              \
        stride_w,                                                              \
        pad_h,                                                                 \
        pad_w,                                                                 \
        dilation_h,                                                            \
        dilation_w);                                                           \
    sycl_kernel_submit(num_wg* wg_size, wg_size, queue, kfn);                  \
  }

template <typename scalar_t, bool is_channels_last, typename index_t = int>
void launch_max_pool2d_backward_kernel(
    scalar_t* gradInput,
    const scalar_t* gradOutput,
    const int64_t* indices,
    index_t numBatch,
    index_t numPlane,
    index_t gradInputSizeH,
    index_t gradInputSizeW,
    index_t gradOutputSizeH,
    index_t gradOutputSizeW,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int64_t gradInputSize = numBatch * numPlane * gradInputSizeH * gradInputSizeW;
  auto out_cf_c_stride = gradOutputSizeH * gradOutputSizeW;
  auto in_cf_c_stride = gradInputSizeH * gradInputSizeW;
  auto out_n_stride = numPlane * out_cf_c_stride;
  auto in_n_stride = numPlane * in_cf_c_stride;

#ifndef XPU_ALLOW_UNDETERMINISTIC
  // [Deterministic Note]

  // By default, we disable the un-derterministic path in this kernel,
  // so that we make sure there will no side-effect with the accuracy.
  // In the future, we will re-enable the un-deterministic path to improve
  // performance.
  //
  // The background of this is that we found this kernel has different behavior
  // with CUDA in alexnet To avoid future problem, we decided to always use
  // deterministic path.

  int vec_size = 1;
  int thread_slots = at::xpu::getDeviceHWThreads();
  int num_sub_wg;
  int64_t wg_size = at::xpu::getDeviceMaxWorkGroupSize();
  int64_t num_wg;
  if constexpr (is_channels_last) {
    for (vec_size = std::min(
             8, memory::can_vectorize_up_to<scalar_t>((char*)gradOutput));
         vec_size >= 1;
         vec_size /= 2) {
      if (numPlane % vec_size != 0) {
        continue;
      }
      num_sub_wg =
          gradInputSize / vec_size / at::xpu::getDeviceMaxSubGroupSize();
      if (2 * num_sub_wg > thread_slots) {
        int total_thread = gradInputSize / vec_size;
        num_wg = (total_thread + wg_size - 1) / wg_size;
        break;
      }
    }
    switch (vec_size) {
      case 8:
        LAUNCH_MAXPOOL_BACKWARD_CHANNEL_LAST_VEC(
            scalar_t,
            8,
            index_t,
            num_wg,
            wg_size,
            queue,
            gradInput,
            gradOutput,
            indices,
            numPlane,
            gradInputSizeH,
            gradInputSizeW,
            gradOutputSizeH,
            gradOutputSizeW,
            gradInputSize,
            out_n_stride,
            in_n_stride,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w);
        return;
      case 4:
        LAUNCH_MAXPOOL_BACKWARD_CHANNEL_LAST_VEC(
            scalar_t,
            4,
            index_t,
            num_wg,
            wg_size,
            queue,
            gradInput,
            gradOutput,
            indices,
            numPlane,
            gradInputSizeH,
            gradInputSizeW,
            gradOutputSizeH,
            gradOutputSizeW,
            gradInputSize,
            out_n_stride,
            in_n_stride,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w);
        return;
      case 2:
        LAUNCH_MAXPOOL_BACKWARD_CHANNEL_LAST_VEC(
            scalar_t,
            2,
            index_t,
            num_wg,
            wg_size,
            queue,
            gradInput,
            gradOutput,
            indices,
            numPlane,
            gradInputSizeH,
            gradInputSizeW,
            gradOutputSizeH,
            gradOutputSizeW,
            gradInputSize,
            out_n_stride,
            in_n_stride,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w);
        return;
      default:
        break;
    };
  }
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  using KernelClass = MaxPool2dBackwardDeterministicKernelFunctor<
      scalar_t,
      accscalar_t,
      is_channels_last,
      index_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      1, gradInputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive);
  cfg.template build<KernelClass>();
  auto kfn = KernelClass(
      gradInput,
      gradOutput,
      indices,
      numPlane,
      gradInputSizeH,
      gradInputSizeW,
      gradOutputSizeH,
      gradOutputSizeW,
      gradInputSize,
      out_cf_c_stride,
      in_cf_c_stride,
      out_n_stride,
      in_n_stride,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      cfg);
  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
#else
  int64_t gradOutputSize =
      numBatch * numPlane * gradOutputSizeH * gradOutputSizeW;
  using KernelClass =
      MaxPool2dBackwardKernelFunctor<scalar_t, is_channels_last, index_t>;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      1, gradOutputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive);
  cfg.template build<KernelClass>();
  auto kfn = KernelClass(
      gradInput,
      gradOutput,
      indices,
      numPlane,
      gradInputSizeH,
      gradInputSizeW,
      gradOutputSizeH,
      gradOutputSizeW,
      gradOutputSize,
      out_cf_c_stride,
      in_cf_c_stride,
      out_n_stride,
      in_n_stride,
      cfg);
  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
#endif
}

void max_pool2d_with_indices_kernel(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& output,
    const Tensor& indices) {
  TensorArg output_arg{output, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{input_, "input_", 3};

  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});
  if (output.numel() == 0) {
    return;
  }

  if (max_pool_optimized::bmg_device() &&
      max_pool_optimized::forward(
          input_, output, indices, kernel_size, stride, padding, dilation)) {
    return;
  }

  auto memory_format = input_.suggest_memory_format();
  Tensor input = input_.contiguous(memory_format);

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

  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputHeight = output.size(-2);
  const int64_t outputWidth = output.size(-1);

  const int64_t in_stride_n = input_.ndimension() == 4 ? input.stride(-4) : 0;
  const int64_t in_stride_c = input.stride(-3);
  const int64_t in_stride_h = input.stride(-2);
  const int64_t in_stride_w = input.stride(-1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "max_pool2d_xpu", [&] {
        switch (memory_format) {
          case MemoryFormat::ChannelsLast: {
            bool use_int32 = can_use_int32_nhwc(
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                in_stride_n,
                in_stride_c,
                in_stride_h,
                in_stride_w);
            if (use_int32) {
              launch_max_pool2d_kernel<scalar_t, true, int32_t>(
                  output.mutable_data_ptr<scalar_t>(),
                  indices.mutable_data_ptr<int64_t>(),
                  input.const_data_ptr<scalar_t>(),
                  static_cast<int32_t>(nbatch),
                  static_cast<int32_t>(nInputPlane),
                  static_cast<int32_t>(inputHeight),
                  static_cast<int32_t>(inputWidth),
                  static_cast<int32_t>(outputHeight),
                  static_cast<int32_t>(outputWidth),
                  kH,
                  kW,
                  dH,
                  dW,
                  padH,
                  padW,
                  dilationH,
                  dilationW);
            } else {
              launch_max_pool2d_kernel<scalar_t, true, int64_t>(
                  output.mutable_data_ptr<scalar_t>(),
                  indices.mutable_data_ptr<int64_t>(),
                  input.const_data_ptr<scalar_t>(),
                  nbatch,
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
                  dilationH,
                  dilationW);
            }
            break;
          }
          case MemoryFormat::Contiguous: {
            bool use_int32 = can_use_int32_nchw(
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth);
            if (use_int32) {
              launch_max_pool2d_kernel<scalar_t, false, int32_t>(
                  output.mutable_data_ptr<scalar_t>(),
                  indices.mutable_data_ptr<int64_t>(),
                  input.const_data_ptr<scalar_t>(),
                  static_cast<int32_t>(nbatch),
                  static_cast<int32_t>(nInputPlane),
                  static_cast<int32_t>(inputHeight),
                  static_cast<int32_t>(inputWidth),
                  static_cast<int32_t>(outputHeight),
                  static_cast<int32_t>(outputWidth),
                  kH,
                  kW,
                  dH,
                  dW,
                  padH,
                  padW,
                  dilationH,
                  dilationW);
            } else {
              launch_max_pool2d_kernel<scalar_t, false, int64_t>(
                  output.mutable_data_ptr<scalar_t>(),
                  indices.mutable_data_ptr<int64_t>(),
                  input.const_data_ptr<scalar_t>(),
                  nbatch,
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
                  dilationH,
                  dilationW);
            }
            break;
          }
          default:
            TORCH_CHECK(
                false,
                "Unsupported memory format. Supports only ChannelsLast, Contiguous");
        }
      });
}

void max_pool2d_with_indices_backward_kernel(
    const Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input_,
    const Tensor& indices_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TensorArg gradInput_arg{gradInput, "gradInput", 1};
  TensorArg gradOutput_arg{gradOutput_, "gradOutput", 2};
  TensorArg input_arg{input_, "input", 3};
  TensorArg indices_arg{indices_, "indices", 4};
  checkAllSameGPU(
      __func__, {gradInput_arg, gradOutput_arg, input_arg, indices_arg});

  if (max_pool_optimized::bmg_device() &&
      max_pool_optimized::backward(
          input_,
          gradOutput_,
          indices_,
          gradInput,
          kernel_size,
          stride,
          padding,
          dilation)) {
    return;
  }

  auto memory_format = input_.suggest_memory_format();
  Tensor input = input_.contiguous(memory_format);
  Tensor gradOutput = gradOutput_.contiguous(memory_format);
  Tensor indices = indices_.contiguous(memory_format);
  gradInput.zero_();

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
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  int64_t outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  int64_t outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  const int64_t in_stride_n = input.ndimension() == 4 ? input.stride(-4) : 0;
  const int64_t in_stride_c = input.stride(-3);
  const int64_t in_stride_h = input.stride(-2);
  const int64_t in_stride_w = input.stride(-1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gradOutput.scalar_type(),
      "max_pool2d_backward_xpu",
      [&] {
        switch (memory_format) {
          case at::MemoryFormat::ChannelsLast: {
            bool use_int32 = can_use_int32_nhwc(
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                in_stride_n,
                in_stride_c,
                in_stride_h,
                in_stride_w);
            if (use_int32) {
              launch_max_pool2d_backward_kernel<scalar_t, true, int32_t>(
                  gradInput.mutable_data_ptr<scalar_t>(),
                  gradOutput.const_data_ptr<scalar_t>(),
                  indices.const_data_ptr<int64_t>(),
                  static_cast<int32_t>(nbatch),
                  static_cast<int32_t>(nInputPlane),
                  static_cast<int32_t>(inputHeight),
                  static_cast<int32_t>(inputWidth),
                  static_cast<int32_t>(outputHeight),
                  static_cast<int32_t>(outputWidth),
                  kH,
                  kW,
                  dH,
                  dW,
                  padH,
                  padW,
                  dilationH,
                  dilationW);
            } else {
              launch_max_pool2d_backward_kernel<scalar_t, true, int64_t>(
                  gradInput.mutable_data_ptr<scalar_t>(),
                  gradOutput.const_data_ptr<scalar_t>(),
                  indices.const_data_ptr<int64_t>(),
                  nbatch,
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
                  dilationH,
                  dilationW);
            }
            break;
          }
          case at::MemoryFormat::Contiguous: {
            bool use_int32 = can_use_int32_nchw(
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth);
            if (use_int32) {
              launch_max_pool2d_backward_kernel<scalar_t, false, int32_t>(
                  gradInput.mutable_data_ptr<scalar_t>(),
                  gradOutput.const_data_ptr<scalar_t>(),
                  indices.const_data_ptr<int64_t>(),
                  static_cast<int32_t>(nbatch),
                  static_cast<int32_t>(nInputPlane),
                  static_cast<int32_t>(inputHeight),
                  static_cast<int32_t>(inputWidth),
                  static_cast<int32_t>(outputHeight),
                  static_cast<int32_t>(outputWidth),
                  kH,
                  kW,
                  dH,
                  dW,
                  padH,
                  padW,
                  dilationH,
                  dilationW);
            } else {
              launch_max_pool2d_backward_kernel<scalar_t, false, int64_t>(
                  gradInput.mutable_data_ptr<scalar_t>(),
                  gradOutput.const_data_ptr<scalar_t>(),
                  indices.const_data_ptr<int64_t>(),
                  nbatch,
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
                  dilationH,
                  dilationW);
            }
            break;
          }
          default:
            TORCH_CHECK(
                false,
                "Unsupported memory format. Supports only ChannelsLast, Contiguous");
        }
      });
}

} // namespace at::native::xpu
#undef LAUNCH_MAXPOOL_CHANNEL_LAST_VEC
// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
