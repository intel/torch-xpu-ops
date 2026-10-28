/*
 * Copyright 2020-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>
#include <climits>
namespace at::native::xpu::bmg_maxpool {
inline int compact_index(int64_t value) {
  return value >= 0 && value <= INT_MAX ? int(value) : -1;
}
struct Div {
  uint32_t d, m;
  Div(uint32_t v) : d(v), m(v == 1 ? 0 : uint32_t((uint64_t(1) << 32) / v)) {}
  uint32_t div(uint32_t x) const {
    if (d == 1)
      return x;
    uint32_t q = sycl::mul_hi(x, m);
    return q + (x - q * d >= d);
  }
};
template <int K, int S, int P>
struct Forward {
  const uint32_t* x;
  uint32_t* y;
  int64_t* idx;
  int H, W, OH, OW, C, count;
  Div dc, dw, dh;
  void operator()(sycl::nd_item<1> it) const {
    for (uint32_t i = it.get_global_linear_id(); i < count;
         i += it.get_global_range(0)) {
      auto pix = dc.div(i), c = i - pix * C, nh = dw.div(pix),
           ox = pix - nh * OW, n = dh.div(nh), oy = nh - n * OH;
      int h0 = int(oy) * S - P, w0 = int(ox) * S - P;
      sycl::float2 best(-INFINITY);
      sycl::vec<int, 2> index(sycl::max(h0, 0) * W + sycl::max(w0, 0));
#pragma unroll
      for (int h = 0; h < K; ++h) {
#pragma unroll
        for (int w = 0; w < K; ++w) {
          int ih = h0 + h, iw = w0 + w;
          if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            auto v = sycl::vec<uint32_t, 1>(x[((n * H + ih) * W + iw) * C + c])
                         .template as<sycl::half2>()
                         .template convert<float>();
#pragma unroll
            for (int j = 0; j < 2; ++j)
              if (v[j] > best[j] || sycl::isnan(v[j])) {
                best[j] = v[j];
                index[j] = ih * W + iw;
              }
          }
        }
      }
      y[i] = best.template convert<sycl::half>()
                 .template as<sycl::vec<uint32_t, 1>>()[0];
      *reinterpret_cast<sycl::vec<int64_t, 2>*>(idx + int64_t(i) * 2) =
          index.template convert<int64_t>();
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
      sycl::float2 best(-INFINITY);
      sycl::int2 bi(0);
#pragma unroll
      for (int xx = 0; xx < 16; ++xx)
        raw[xx] = x[((n * 12 + row) * 16 + xx) * C + cb * 16 + c];
#pragma unroll
      for (int xx = 0; xx < 16; ++xx) {
        auto v = sycl::vec<uint32_t, 1>(raw[xx])
                     .template as<sycl::half2>()
                     .template convert<float>();
#pragma unroll
        for (int j = 0; j < 2; ++j)
          if (xx % K == 0 || v[j] > best[j] || sycl::isnan(v[j])) {
            best[j] = v[j];
            bi[j] = xx;
          }
        pv[xx] = best.template convert<sycl::half>()
                     .template as<sycl::vec<uint32_t, 1>>()[0];
        pi[xx] = bi[0] | (bi[1] << 8);
        if (xx >= K / 2 && xx < K) {
          int dest = (row * 16 + xx - K / 2) * 16 + c;
          smem[dest] = pv[xx];
          smem[3072 + dest] = pi[xx];
        }
      }
      best = sycl::float2(-INFINITY);
      bi = sycl::int2(15);
#pragma unroll
      for (int xx = 15; xx >= 0; --xx) {
        auto v = sycl::vec<uint32_t, 1>(raw[xx])
                     .template as<sycl::half2>()
                     .template convert<float>();
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
            auto bv = sycl::vec<uint32_t, 1>(pv[hi])
                          .template as<sycl::half2>()
                          .template convert<float>();
            sycl::int2 bj(pi[hi] & 255, pi[hi] >> 8);
#pragma unroll
            for (int j = 0; j < 2; ++j)
              if (bv[j] > vv[j] || sycl::isnan(bv[j])) {
                vv[j] = bv[j];
                ii[j] = bj[j];
              }
          }
          int dest = (row * 16 + xx + K / 2) * 16 + c;
          smem[dest] = vv.template convert<sycl::half>()
                           .template as<sycl::vec<uint32_t, 1>>()[0];
          smem[3072 + dest] = ii[0] | (ii[1] << 8);
        }
      }
    }
    it.barrier(sycl::access::fence_space::local_space);
    sycl::float2 hv[12];
    sycl::int2 hi[12];
#pragma unroll
    for (int h = 0; h < 12; ++h) {
      int a = (h * 16 + ox) * 16 + c;
      hv[h] = sycl::vec<uint32_t, 1>(smem[a])
                  .template as<sycl::half2>()
                  .template convert<float>();
      uint32_t q = smem[3072 + a];
      hi[h] = sycl::int2(h * 16 + (q & 255), h * 16 + (q >> 8));
    }
    if constexpr (K == 13) {
      sycl::float2 best(-INFINITY);
      sycl::int2 bi = hi[0];
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
          y[dest] = best.template convert<sycl::half>()
                        .template as<sycl::vec<uint32_t, 1>>()[0];
          *reinterpret_cast<sycl::vec<int64_t, 2>*>(idx + int64_t(dest) * 2) =
              bi.template convert<int64_t>();
        }
      }
      {
        int dest = ((n * 12 + 6) * 16 + ox) * C + cb * 16 + c;
        y[dest] = best.template convert<sycl::half>()
                      .template as<sycl::vec<uint32_t, 1>>()[0];
        *reinterpret_cast<sycl::vec<int64_t, 2>*>(idx + int64_t(dest) * 2) =
            bi.template convert<int64_t>();
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
          y[dest] = best.template convert<sycl::half>()
                        .template as<sycl::vec<uint32_t, 1>>()[0];
          *reinterpret_cast<sycl::vec<int64_t, 2>*>(idx + int64_t(dest) * 2) =
              bi.template convert<int64_t>();
        }
      }
    } else {
#pragma unroll
      for (int h = 0; h < 12; ++h) {
        sycl::float2 best(-INFINITY);
        sycl::vec<int, 2> index = hi[sycl::max(0, h - K / 2)];
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
        y[dest] = best.template convert<sycl::half>()
                      .template as<sycl::vec<uint32_t, 1>>()[0];
        *reinterpret_cast<sycl::vec<int64_t, 2>*>(idx + dest * 2) =
            index.template convert<int64_t>();
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
      sycl::float2 best(-INFINITY);
      sycl::int2 bi(
          sycl::max(oy * 2 - 1, 0) * 16 + sycl::max(ox * 2 - 1, 0),
          sycl::max(oy * 2 - 1, 0) * 16 + ox * 2 + 1);
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
      y[i] = best.template convert<sycl::half>()
                 .template as<sycl::vec<uint32_t, 1>>()[0];
      *reinterpret_cast<sycl::vec<int64_t, 2>*>(idx + int64_t(i) * 2) =
          bi.template convert<int64_t>();
    }
  }
};

template <int K, int P, int CB, int XB>
struct BlockBackward {
  const uint32_t* go;
  const int64_t* idx;
  uint32_t* gi;
  int H, W, OH, OW, C, CS, XT, YT, CT;
  Div dct, dxt, dyt;
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
        sycl::float2 acc[4] = {
            sycl::float2(0), sycl::float2(0), sycl::float2(0), sycl::float2(0)};
#pragma unroll
        for (int hh = 0; hh < NH; ++hh) {
#pragma unroll
          for (int ww = 0; ww < NH; ++ww) {
            int ph = yb + DELTA + hh, pw = pwbase + DELTA + ww;
            if (ph >= 0 && ph < OH && pw >= 0 && pw < OW) {
              auto val = sycl::vec<uint32_t, 1>(
                             go[((n * OH + ph) * OW + pw) * CS + (cb + c) / 2])
                             .template as<sycl::half2>()
                             .template convert<float>();
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
                  acc[dy * 2 + dx]
                      .template convert<sycl::half, sycl::rounding_mode::rte>()
                      .template as<sycl::vec<uint32_t, 1>>()[0];
        }
      }
    }
  }
};
template <int K, int P, int CB, int XB>
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
        BlockBackward<K, P, CB, XB>{
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
            Div(CT),
            Div(XT),
            Div(YT),
            li});
  });
}
template <int K, int P>
void block_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  if (x.size(3) <= 16)
    block_launch_impl<K, P, 64, 8>(x, v, idx, out);
  else if (x.size(3) <= 32)
    block_launch_impl<K, P, 64, 16>(x, v, idx, out);
  else
    block_launch_impl<K, P, 64, 32>(x, v, idx, out);
}

template <int K, int P, int CB, int XB>
struct BlockBackward4 {
  const uint64_t* go;
  const int64_t* idx;
  uint64_t* gi;
  int H, W, OH, OW, C, CS, XT, YT, CT;
  Div dct, dxt, dyt;
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
            (cb + c < C * 4 && ph >= 0 && ph < OH && pw >= 0 && pw < OW)
            ? compact_index(idx[((n * C * 4 + cb + c) * OH + ph) * OW + pw])
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
              (cb + c < C * 4 && ph >= 0 && ph < OH && pw >= 0 && pw < OW)
              ? compact_index(idx[((n * C * 4 + cb + c) * OH + ph) * OW + pw])
              : -1;
        }
      }
    }
    it.barrier(sycl::access::fence_space::local_space);
    for (int j = it.get_local_id(0); j < XB * CB / 4; j += 256) {
      int c = j % (CB / 4) * 4, xx = j / (CB / 4), pwbase = xb + xx,
          iy = yb * 2, ix = pwbase * 2;
      if (cb + c < C * 4 && ix < W) {
        sycl::float4 acc[4] = {
            sycl::float4(0), sycl::float4(0), sycl::float4(0), sycl::float4(0)};
#pragma unroll
        for (int hh = 0; hh < NH; ++hh) {
#pragma unroll
          for (int ww = 0; ww < NH; ++ww) {
            int ph = yb + DELTA + hh, pw = pwbase + DELTA + ww;
            if (ph >= 0 && ph < OH && pw >= 0 && pw < OW) {
              auto val = sycl::vec<uint64_t, 1>(
                             go[((n * OH + ph) * OW + pw) * CS + (cb + c) / 4])
                             .template as<sycl::half4>()
                             .template convert<float>();
              int a = (hh * NW + xx + ww) * (CB + 1) + c;
              sycl::int4 ids(li[a], li[a + 1], li[a + 2], li[a + 3]);
#pragma unroll
              for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
                for (int dx = 0; dx < 2; ++dx) {
                  if (K == 2 ||
                      ((P == 1 ? (hh == 0 || dy == 1) : (hh == 1 || dy == 0)) &&
                       (P == 1 ? (ww == 0 || dx == 1)
                               : (ww == 1 || dx == 0)))) {
                    int target = (iy + dy) * W + ix + dx;
                    acc[dy * 2 + dx] += sycl::select(
                        sycl::float4(0), val, ids == sycl::int4(target));
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
              gi[((n * H + iy + dy) * W + ix + dx) * C + (cb + c) / 4] =
                  acc[dy * 2 + dx]
                      .template convert<sycl::half, sycl::rounding_mode::rte>()
                      .template as<sycl::vec<uint64_t, 1>>()[0];
        }
      }
    }
  }
};
template <int K, int P, int CB, int XB>
void block4_launch_impl(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  int H = x.size(2), W = x.size(3), C = x.size(1) / 4, OH = v.size(2),
      OW = v.size(3), XT = (W + 2 * XB - 1) / (2 * XB), YT = (H + 1) / 2,
      CT = (C * 4 + CB - 1) / CB;
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int, 1> li((K == 2 ? XB : 2 * (XB + 1)) * (CB + 1), h);
    h.parallel_for(
        sycl::nd_range<1>(x.size(0) * XT * YT * CT * 256, 256),
        BlockBackward4<K, P, CB, XB>{
            (const uint64_t*)v.data_ptr(),
            idx.data_ptr<int64_t>(),
            (uint64_t*)out.data_ptr(),
            H,
            W,
            OH,
            OW,
            C,
            int(v.stride(3) / 4),
            XT,
            YT,
            CT,
            Div(CT),
            Div(XT),
            Div(YT),
            li});
  });
}
template <int K, int P>
void block4_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  if (x.size(1) <= 32)
    block4_launch_impl<K, P, 32, 32>(x, v, idx, out);
  else if (x.size(3) <= 16)
    block4_launch_impl<K, P, 64, 8>(x, v, idx, out);
  else if (x.size(3) <= 32)
    block4_launch_impl<K, P, 64, 16>(x, v, idx, out);
  else
    block4_launch_impl<K, P, 64, 32>(x, v, idx, out);
}

struct EDiv {
  uint32_t d, m;
  EDiv(uint32_t v) : d(v), m(v == 1 ? 0 : uint32_t((uint64_t(1) << 32) / v)) {}
  uint32_t div(uint32_t x) const SYCL_ESIMD_FUNCTION {
    if (d == 1)
      return x;
    uint32_t q = (uint64_t(x) * m) >> 32;
    return q + (x - q * d >= d);
  }
};
SYCL_ESIMD_FUNCTION inline void paired_index_load(
    const uint64_t* src,
    int C,
    int HW,
    int cb,
    int p0,
    sycl::ext::intel::esimd::simd<uint32_t, 128>& lo,
    sycl::ext::intel::esimd::simd<uint32_t, 128>& high) {
  namespace e = sycl::ext::intel::esimd;
#pragma unroll
  for (int parity = 0; parity < 2; ++parity) {
    auto raw = e::load_2d<uint32_t, 8, 16, 1, true>(
        (const uint32_t*)src,
        HW * 16 - 1,
        C / 2 - 1,
        HW * 16 - 1,
        (p0 + parity * HW) * 2,
        cb / 2);
    lo.template bit_cast_view<uint32_t, 4, 32>().template select<4, 1, 16, 2>(
        0, parity) = raw.template bit_cast_view<uint32_t, 8, 16>()
                         .template select<4, 2, 16, 1>(0, 0);
    high.template bit_cast_view<uint32_t, 4, 32>().template select<4, 1, 16, 2>(
        0, parity) = raw.template bit_cast_view<uint32_t, 8, 16>()
                         .template select<4, 2, 16, 1>(1, 0);
  }
}

struct EBackward2 {
  const uint32_t* idx;
  const sycl::half* go;
  sycl::half* gi;
  int H, W, OH, OW, C, CS, CT, XT, count;
  EDiv dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = dc.div(id), c = (id - q * CT) * 32, r = dx.div(q),
             px = (q - r * XT) * 4, n = dy.div(r), ph = r - n * OH;
    int HW = OH * OW, pbase = ph * OW + px;
    e::simd<uint32_t, 128> indices, high;
    if (HW % 2 == 0) {
      auto raw = e::load_2d<uint32_t, 8, 32, 1, true>(
          idx + n * C * HW * 2, HW * 8 - 1, C - 1, HW * 8 - 1, pbase * 2, c);
      indices = raw.template bit_cast_view<uint32_t, 8, 32>()
                    .template select<4, 2, 32, 1>(0, 0)
                    .read();
      high = raw.template bit_cast_view<uint32_t, 8, 32>()
                 .template select<4, 2, 32, 1>(1, 0)
                 .read();
    } else
      paired_index_load(
          (const uint64_t*)idx + n * C * HW, C, HW, c, pbase, indices, high);
    indices.merge(uint32_t(-1), high != 0);

    auto val = e::load_2d<sycl::half, 32, 4>(
        go + n * HW * CS, C * 2 - 1, HW - 1, CS * 2 - 1, c, pbase);
    e::simd<int, 128> lane(0, 1);
    e::simd<int, 128> target = ph * 2 * W + (px + lane / 32) * 2;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        e::simd<float, 128> sum = val;
        sum += 0.f;
        e::simd<sycl::half, 128> out = sum;
        out.merge(sycl::half(0), indices != target + h * W + w);
        e::store_2d<sycl::half, 32, 4>(
            gi + ((n * H + ph * 2 + h) * W + w) * C,
            C * 2 - 1,
            OW - 1,
            C * 4 - 1,
            c,
            px,
            out);
      }
    }
  }
};
void esimd2_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  int H = x.size(2), W = x.size(3), OH = v.size(2), OW = v.size(3),
      C = x.size(1), CT = (C + 31) / 32, XT = (OW + 3) / 4,
      count = x.size(0) * OH * CT * XT;
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      EBackward2{
          (const uint32_t*)idx.data_ptr(),
          (const sycl::half*)v.data_ptr(),
          (sycl::half*)out.data_ptr(),
          H,
          W,
          OH,
          OW,
          C,
          int(v.stride(3)),
          CT,
          XT,
          count,
          EDiv(CT),
          EDiv(XT),
          EDiv(OH)});
}

template <int V, bool Block = false>
struct EForwardS1 {
  const sycl::half* x;
  sycl::half* y;
  uint64_t* idx;
  int H, W, C, CT, XT, YT, count;
  EDiv dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t i = it.get_global_linear_id();
    if (i >= count)
      return;
    uint32_t q = dc.div(i), c = (i - q * CT) * V, r = dx.div(q),
             ox = (q - r * XT) * 2, n = dy.div(r), oy = (r - n * YT) * 2;
    e::simd<sycl::half, V> v[16], best[4];
    e::simd<uint16_t, V> bi[4];
    e::simd<uint32_t, V> lane(0, 1);
    if constexpr (Block) {
#pragma unroll
      for (int h = 0; h < 4; ++h) {
#pragma unroll
        for (int w = 0; w < 4; ++w)
          v[h * 4 + w] = sycl::half(-INFINITY);
        int ih = int(oy) + h - 1;
        if (ih >= 0 && ih < H) {
#pragma unroll
          for (int cc = 0; cc < V; cc += 32) {
            auto values = e::load_2d<sycl::half, 32, 4>(
                x + (n * H + ih) * W * C,
                C * 2 - 1,
                W - 1,
                C * 2 - 1,
                c + cc,
                int(ox) - 1);
#pragma unroll
            for (int w = 0; w < 4; ++w)
              if (int(ox) + w - 1 >= 0 && int(ox) + w - 1 < W)
                v[h * 4 + w].template select<32, 1>(cc) =
                    values.template select<32, 1>(w * 32);
          }
        }
      }
    } else {
#pragma unroll
      for (int h = 0; h < 4; ++h) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          int ih = int(oy) + h - 1, iw = int(ox) + w - 1;
          v[h * 4 + w] = sycl::half(-INFINITY);
          if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            auto ptr = x + ((n * H + ih) * W + iw) * C + c;
            if (c + V <= C)
              v[h * 4 + w] = e::block_load<sycl::half, V>(
                  ptr, e::properties{e::alignment<4>});
            else
              v[h * 4 + w] = e::gather<sycl::half, V>(
                  ptr, lane * 2, lane < uint32_t(C - c));
          }
        }
      }
    }
#pragma unroll
    for (int dhh = 0; dhh < 2; ++dhh) {
#pragma unroll
      for (int dww = 0; dww < 2; ++dww) {
        best[dhh * 2 + dww] = sycl::half(-INFINITY);
        bi[dhh * 2 + dww] = uint16_t(
            (int(oy) + dhh > 0 ? int(oy) + dhh - 1 : 0) * W +
            (int(ox) + dww > 0 ? int(ox) + dww - 1 : 0));
      }
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) {
#pragma unroll
      for (int w = 0; w < 4; ++w) {
#pragma unroll
        for (int dhh = 0; dhh < 2; ++dhh) {
#pragma unroll
          for (int dww = 0; dww < 2; ++dww) {
            if (oy + dhh < H && ox + dww < W && h >= dhh && h <= dhh + 2 &&
                w >= dww && w <= dww + 2) {
              auto mask = (v[h * 4 + w] > best[dhh * 2 + dww]) |
                  (v[h * 4 + w] != v[h * 4 + w]);
              best[dhh * 2 + dww].merge(v[h * 4 + w], mask);
              bi[dhh * 2 + dww].merge(
                  uint16_t((int(oy) + h - 1) * W + int(ox) + w - 1), mask);
            }
          }
        }
      }
    }
#pragma unroll
    for (int dhh = 0; dhh < 2; ++dhh) {
#pragma unroll
      for (int dww = 0; dww < 2; ++dww) {
        if (oy + dhh < H && ox + dww < W) {
          uint32_t dest = ((n * H + oy + dhh) * W + ox + dww) * C + c;
          e::simd<uint64_t, V> ix = bi[dhh * 2 + dww];
          if (c + V <= C) {
            e::block_store<sycl::half, V>(
                y + dest, best[dhh * 2 + dww], e::properties{e::alignment<4>});
            e::block_store<uint64_t, V>(
                idx + uint64_t(dest), ix, e::properties{e::alignment<16>});
          } else {
            e::simd<uint32_t, (V / 2)> l(0, 1);
            e::simd<uint32_t, (V / 2)> packed =
                best[dhh * 2 + dww].template bit_cast_view<uint32_t>();
            e::scatter<uint32_t, (V / 2)>(
                (uint32_t*)(y + dest),
                l * 4,
                packed,
                l < uint32_t((C - c) / 2));
            e::scatter<uint64_t, V>(
                idx + uint64_t(dest), lane * 8, ix, lane < uint32_t(C - c));
          }
        }
      }
    }
  }
};
template <int V, bool Block = false>
void eforward_s1_impl(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& idx) {
  int H = x.size(2), W = x.size(3), C = x.size(1), CT = (C + V - 1) / V,
      XT = (W + 1) / 2, YT = (H + 1) / 2, count = x.size(0) * CT * XT * YT;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      EForwardS1<V, Block>{
          (const sycl::half*)x.data_ptr(),
          (sycl::half*)y.data_ptr(),
          (uint64_t*)idx.data_ptr(),
          H,
          W,
          C,
          CT,
          XT,
          YT,
          count,
          EDiv(CT),
          EDiv(XT),
          EDiv(YT)});
}

void eforward_s1_launch(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& idx) {
  if (x.size(1) == 432 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 64 == 0)
    eforward_s1_impl<64, true>(x, y, idx);
  else if (x.size(1) % 64 > 0 && x.size(1) % 64 <= 32)
    eforward_s1_impl<32>(x, y, idx);
  else
    eforward_s1_impl<64>(x, y, idx);
}
template <int P>
struct EBackward3 {
  const uint64_t* idx;
  const sycl::half* go;
  sycl::half* gi;
  int H, W, OH, OW, C, CS, CT, XT, YT, count;
  EDiv dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = dc.div(id), c = (id - q * CT) * 32, r = dx.div(q),
             px = (q - r * XT) * 4, n = dy.div(r), yb = r - n * YT;
    int HW = OH * OW;
    constexpr int D = P == 0 ? -1 : 0;
    e::simd<float, 128> acc[4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
      acc[a] = 0;
    e::simd<uint32_t, 128> lane(0, 1),
        target = yb * 2 * W + (px + lane / 32) * 2;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      int ph = int(yb) + D + hh;
      if (ph >= 0 && ph < OH) {
#pragma unroll
        for (int ww = 0; ww < 2; ++ww) {
          int pw = int(px) + D + ww, pbase = ph * OW + pw;
          e::simd<uint32_t, 128> indices;
          if (HW % 2 == 0) {
            auto raw = e::load_2d<uint32_t, 8, 32, 1, true>(
                (const uint32_t*)(idx + n * C * HW),
                HW * 8 - 1,
                C - 1,
                HW * 8 - 1,
                pbase * 2,
                c);
            indices = raw.template bit_cast_view<uint32_t, 8, 32>()
                          .template select<4, 2, 32, 1>(0, 0)
                          .read();
            e::simd<uint32_t, 128> high =
                raw.template bit_cast_view<uint32_t, 8, 32>()
                    .template select<4, 2, 32, 1>(1, 0)
                    .read();
            indices.merge(uint32_t(-1), high != 0);
          } else {
            e::simd<uint32_t, 128> high;
            paired_index_load(idx + n * C * HW, C, HW, c, pbase, indices, high);
            indices.merge(uint32_t(-1), high != 0);
          }

          e::simd<sycl::half, 128> half;
          if (CS % 8 == 0)
            half = e::load_2d<sycl::half, 32, 4>(
                go + n * HW * CS, C * 2 - 1, HW - 1, CS * 2 - 1, c, pbase);
          else {
#pragma unroll
            for (int parity = 0; parity < 2; ++parity) {
              int pos = pbase + parity;
              auto pair = e::load_2d<sycl::half, 32, 2>(
                  go + n * HW * CS,
                  (CS + C) * 2 - 1,
                  HW / 2 - 1,
                  CS * 4 - 1,
                  (pos & 1) * CS + c,
                  pos >> 1);
              half.template bit_cast_view<sycl::half, 4, 32>()
                  .template select<2, 2, 32, 1>(parity, 0) = pair;
            }
          }
          e::simd<float, 128> v = half;
          auto valid = (lane / 32 + pw < uint32_t(OW));
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int w = 0; w < 2; ++w) {
              if ((P == 1 ? (hh == 0 || h == 1) : (hh == 1 || h == 0)) &&
                  (P == 1 ? (ww == 0 || w == 1) : (ww == 1 || w == 0))) {
                e::simd<float, 128> add = v;
                add.merge(0.f, (indices != target + h * W + w) | !valid);
                acc[h * 2 + w] += add;
              }
            }
          }
        }
      }
    }
    e::simd<sycl::half, 128> out[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
      out[j] = acc[j];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (yb * 2 + h < H) {
#pragma unroll
        for (int w = 0; w < 2; ++w)
          e::store_2d<sycl::half, 32, 4>(
              gi + ((n * H + yb * 2 + h) * W + w) * C,
              C * 2 - 1,
              (W + 1 - w) / 2 - 1,
              C * 4 - 1,
              c,
              px,
              out[h * 2 + w]);
      }
    }
  }
};
template <int P>
void eb3_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  int H = x.size(2), W = x.size(3), OH = v.size(2), OW = v.size(3),
      C = x.size(1), CT = (C + 31) / 32, XT = (W + 7) / 8, YT = (H + 1) / 2,
      count = x.size(0) * YT * XT * CT;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      EBackward3<P>{
          (const uint64_t*)idx.data_ptr(),
          (const sycl::half*)v.data_ptr(),
          (sycl::half*)out.data_ptr(),
          H,
          W,
          OH,
          OW,
          C,
          int(v.stride(3)),
          CT,
          XT,
          YT,
          count,
          EDiv(CT),
          EDiv(XT),
          EDiv(YT)});
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
          Div(C),
          Div(OW),
          Div(OH)});
}
template <int K, int S, int PAD, int V>
struct EForward {
  const sycl::half* x;
  sycl::half* y;
  uint64_t* idx;
  int H, W, OH, OW, C, CT, count;
  EDiv dc, dw, dh;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = dc.div(id), c = (id - q * CT) * V, nh = dw.div(q),
             ox = q - nh * OW, n = dh.div(nh), oy = nh - n * OH;
    int h0 = int(oy) * S - PAD, w0 = int(ox) * S - PAD;
    e::simd<sycl::half, V> vals[K * K];
    e::simd<uint32_t, V> lane(0, 1);
#pragma unroll
    for (int h = 0; h < K; ++h) {
#pragma unroll
      for (int w = 0; w < K; ++w) {
        int ih = h0 + h, iw = w0 + w;
        vals[h * K + w] = sycl::half(-INFINITY);
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          auto ptr = x + ((n * H + ih) * W + iw) * C + c;
          if (c + V <= C)
            vals[h * K + w] = e::block_load<sycl::half, V>(
                ptr, e::properties{e::alignment<4>});
          else
            vals[h * K + w] =
                e::gather<sycl::half, V>(ptr, lane * 2, lane < uint32_t(C - c));
        }
      }
    }
    e::simd<sycl::half, V> best = sycl::half(-INFINITY);
    e::simd<uint32_t, V> bi = (h0 > 0 ? h0 : 0) * W + (w0 > 0 ? w0 : 0);
#pragma unroll
    for (int h = 0; h < K; ++h) {
#pragma unroll
      for (int w = 0; w < K; ++w) {
        auto mask =
            (vals[h * K + w] > best) | (vals[h * K + w] != vals[h * K + w]);
        best.merge(vals[h * K + w], mask);
        bi.merge(uint32_t((h0 + h) * W + w0 + w), mask);
      }
    }
    uint32_t dest = ((n * OH + oy) * OW + ox) * C + c;
    e::simd<uint64_t, V> ix = bi;
    if (c + V <= C) {
      e::block_store<sycl::half, V>(
          y + dest, best, e::properties{e::alignment<4>});
      e::block_store<uint64_t, V>(
          idx + uint64_t(dest), ix, e::properties{e::alignment<16>});
    } else {
      e::simd<uint32_t, V / 2> l(0, 1),
          packed = best.template bit_cast_view<uint32_t>();
      e::scatter<uint32_t, V / 2>(
          (uint32_t*)(y + dest), l * 4, packed, l < uint32_t((C - c) / 2));
      e::scatter<uint64_t, V>(
          idx + uint64_t(dest), lane * 8, ix, lane < uint32_t(C - c));
    }
  }
};
template <int K, int S, int P, int V>
void ef_launch_impl(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx) {
  int C = x.size(1), CT = (C + V - 1) / V,
      count = x.size(0) * v.size(2) * v.size(3) * CT;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      EForward<K, S, P, V>{
          (const sycl::half*)x.data_ptr(),
          (sycl::half*)v.data_ptr(),
          (uint64_t*)idx.data_ptr(),
          int(x.size(2)),
          int(x.size(3)),
          int(v.size(2)),
          int(v.size(3)),
          C,
          CT,
          count,
          EDiv(CT),
          EDiv(v.size(3)),
          EDiv(v.size(2))});
}
template <int K, int S, int P>
void ef_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx) {
  if (x.size(1) < 32)
    ef_launch_impl<K, S, P, 32>(x, v, idx);
  else
    ef_launch_impl<K, S, P, 64>(x, v, idx);
}

template <int P>
struct EBackwardTriple {
  const uint64_t* idx;
  const sycl::half* go;
  sycl::half* gi;
  int H, W, OH, OW, C, CS, CT, XT, YT, count;
  EDiv dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = dc.div(id), c = (id - q * CT) * 32, r = dx.div(q),
             px = (q - r * XT) * 3, n = dy.div(r), yb = r - n * YT;
    int HW = OH * OW;
    constexpr int D = P == 0 ? -1 : 0;
    e::simd<float, 96> acc[4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
      acc[a] = 0;
    e::simd<uint32_t, 96> lane(0, 1),
        target = yb * 2 * W + (px + lane / 32) * 2;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      int ph = int(yb) + D + hh;
      if (ph >= 0 && ph < OH) {
        int pw = int(px) + D, pbase = ph * OW + pw;
        e::simd<uint32_t, 128> indices, high;
        if (HW % 2 == 0) {
          auto raw = e::load_2d<uint32_t, 8, 32, 1, true>(
              (const uint32_t*)(idx + n * C * HW),
              HW * 8 - 1,
              C - 1,
              HW * 8 - 1,
              pbase * 2,
              c);
          indices = raw.template bit_cast_view<uint32_t, 8, 32>()
                        .template select<4, 2, 32, 1>(0, 0)
                        .read();
          high = raw.template bit_cast_view<uint32_t, 8, 32>()
                     .template select<4, 2, 32, 1>(1, 0)
                     .read();
        } else
          paired_index_load(idx + n * C * HW, C, HW, c, pbase, indices, high);
        indices.merge(uint32_t(-1), high != 0);
        auto half = e::load_2d<sycl::half, 32, 4>(
            go + n * HW * CS, C * 2 - 1, HW - 1, CS * 2 - 1, c, pbase);
#pragma unroll
        for (int ww = 0; ww < 2; ++ww) {
          e::simd<uint32_t, 96> ix = indices.template select<96, 1>(ww * 32);
          e::simd<float, 96> v = half.template select<96, 1>(ww * 32);
          auto valid = (lane / 32 + pw + ww < uint32_t(OW));
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int w = 0; w < 2; ++w) {
              if ((P == 1 ? (hh == 0 || h == 1) : (hh == 1 || h == 0)) &&
                  (P == 1 ? (ww == 0 || w == 1) : (ww == 1 || w == 0))) {
                e::simd<float, 96> add = v;
                add.merge(0.f, (ix != target + h * W + w) | !valid);
                acc[h * 2 + w] += add;
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
          e::simd<sycl::half, 96> v = acc[h * 2 + w];
          e::store_2d<sycl::half, 32, 3>(
              gi + ((n * H + yb * 2 + h) * W + w) * C,
              C * 2 - 1,
              (W + 1 - w) / 2 - 1,
              C * 4 - 1,
              c,
              px,
              v);
        }
      }
    }
  }
};
template <int P>
void eb3t_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  int H = x.size(2), W = x.size(3), OH = v.size(2), OW = v.size(3),
      C = x.size(1), CT = (C + 31) / 32, XT = (W + 5) / 6, YT = (H + 1) / 2,
      count = x.size(0) * YT * XT * CT;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      EBackwardTriple<P>{
          (const uint64_t*)idx.data_ptr(),
          (const sycl::half*)v.data_ptr(),
          (sycl::half*)out.data_ptr(),
          H,
          W,
          OH,
          OW,
          C,
          int(v.stride(3)),
          CT,
          XT,
          YT,
          count,
          EDiv(CT),
          EDiv(XT),
          EDiv(YT)});
}

template <int K, int PAD, int V, int CF = 0>
struct EReuseSeven {
  const uint64_t* idx;
  const sycl::half* go;
  sycl::half* gi;
  int H, W, OH, OW, C, CS, CT, XT, YT, count, span;
  EDiv dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    constexpr int P = 7, NH = K == 2 ? 1 : 2, SW = 8, NG = P + NH - 1,
                  D = (K == 3 && PAD == 0) ? -1 : 0;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = dc.div(id), cb = (id - q * CT) * V, r = dx.div(q),
             px = (q - r * XT) * P, n = dy.div(r), yb = r - n * YT;
    int HW = OH * OW;
    e::simd<float, 448> acc_overlap = 0;
    e::simd<sycl::half, 448> acc_half[3];
#pragma unroll
    for (int j = 0; j < 3; ++j)
      acc_half[j] = sycl::half(0);
    e::simd<uint32_t, 64> lane(0, 1);
#pragma unroll
    for (int hh = 0; hh < NH; ++hh) {
      int ph = int(yb) + D + hh;
      if (ph >= 0 && ph < OH) {
        e::simd<uint16_t, V * SW> indices;
#pragma unroll
        for (int block = 0; block < 2; ++block) {
#pragma unroll
          for (int cb = 0; cb < 64; cb += 32) {
            e::simd<uint32_t, 128> lo, high;
            paired_index_load(
                idx + n * C * HW,
                C,
                HW,
                cb,
                ph * OW + px + D + block * 4,
                lo,
                high);
            e::simd<uint16_t, 128> compact = lo;
            compact.merge(uint16_t(65535), (lo >= 65535) | (high != 0));
#pragma unroll
            for (int pp = 0; pp < 4; ++pp)
              indices.template select<32, 1>((block * 4 + pp) * 64 + cb) =
                  compact.template select<32, 1>(pp * 32);
          }
        }
        e::simd<sycl::half, V> values[NG];
#pragma unroll
        for (int j = 0; j < NG; ++j) {
          int pw = int(px) + D + j;
          values[j] = 0;
          if (pw >= 0 && pw < OW) {
            int off = ((n * OH + ph) * OW + pw) * CS + cb;
            if (off + V <= span)
              values[j] = e::block_load<sycl::half, V>(
                  go + off, e::properties{e::alignment<4>});
            else {
              e::simd<uint32_t, V> c(0, 1);
              values[j] = e::gather<sycl::half, V>(
                  go + off, c * 2, c < uint32_t(C - cb));
            }
          }
        }
#pragma unroll
        for (int p = 0; p < P; ++p) {
#pragma unroll
          for (int ww = 0; ww < NH; ++ww) {
            e::simd<uint16_t, V> ix =
                indices.template select<V, 1>((p + ww) * V);
            e::simd<sycl::half, V> v = values[p + ww];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
              for (int w = 0; w < 2; ++w) {
                if (K == 2 ||
                    ((PAD == 1 ? (hh == 0 || h == 1) : (hh == 1 || h == 0)) &&
                     (PAD == 1 ? (ww == 0 || w == 1) : (ww == 1 || w == 0)))) {
                  e::simd<sycl::half, V> add = v;
                  add.merge(
                      sycl::half(0),
                      ix != uint16_t((yb * 2 + h) * W + (px + p) * 2 + w));
                  if (h * 2 + w == (PAD == 1 ? 3 : 0)) {
                    e::simd<float, V> val = add;
                    acc_overlap.template select<V, 1>(p * V) += val;
                  } else
                    acc_half[(PAD == 1 ? h * 2 + w : h * 2 + w - 1)]
                        .template select<V, 1>(p * V) += add;
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
        if constexpr (CF > 0) {
          constexpr int WORDS = P * CF, PW = (WORDS + 15) / 16 * 16;
          e::simd<sycl::half, PW * 2> row = 0;
#pragma unroll
          for (int p = 0; p < P; ++p) {
#pragma unroll
            for (int w = 0; w < 2; ++w) {
              e::simd<sycl::half, CF> val;
              if (h * 2 + w == (PAD == 1 ? 3 : 0))
                val = acc_overlap.template select<CF, 1>(p * V);
              else
                val = acc_half[(PAD == 1 ? h * 2 + w : h * 2 + w - 1)]
                          .template select<CF, 1>(p * V);
              row.template select<CF, 1>(p * CF * 2 + w * CF) = val;
            }
          }
          e::simd<uint32_t, PW> packed = row.template bit_cast_view<uint32_t>();
          auto target =
              (uint32_t*)(gi + ((n * H + yb * 2 + h) * W + px * 2) * C);
          int valid =
              (int(P * 2) < W - int(px * 2) ? int(P * 2) : W - int(px * 2)) *
              CF / 2;
#pragma unroll
          for (int j = 0; j < WORDS / 128; ++j) {
            auto v = packed.template select<128, 1>(j * 128).read();
            if (valid >= j * 128 + 128)
              e::block_store<uint32_t, 128>(
                  target + j * 128, v, e::properties{e::alignment<4>});
            else {
              e::simd<uint32_t, 128> l(0, 1);
              e::scatter<uint32_t, 128>(
                  target + j * 128, l * 4, v, l + j * 128 < uint32_t(valid));
            }
          }
#pragma unroll
          for (int j = WORDS / 128 * 128; j < PW; j += 16) {
            auto v = packed.template select<16, 1>(j).read();
            if (valid >= j + 16)
              e::block_store<uint32_t, 16>(
                  target + j, v, e::properties{e::alignment<4>});
            else {
              e::simd<uint32_t, 16> l(0, 1);
              e::scatter<uint32_t, 16>(
                  target + j, l * 4, v, l + j < uint32_t(valid));
            }
          }
        } else {
#pragma unroll
          for (int p = 0; p < P; ++p) {
#pragma unroll
            for (int w = 0; w < 2; ++w)
              if ((px + p) * 2 + w < W) {
                e::simd<sycl::half, V> v;
                if (h * 2 + w == (PAD == 1 ? 3 : 0))
                  v = acc_overlap.template select<V, 1>(p * V);
                else
                  v = acc_half[(PAD == 1 ? h * 2 + w : h * 2 + w - 1)]
                          .template select<V, 1>(p * V);
                e::block_store<sycl::half, V>(
                    gi + ((n * H + yb * 2 + h) * W + (px + p) * 2 + w) * C + cb,
                    v,
                    e::properties{e::alignment<16>});
              }
          }
        }
      }
    }
  }
};
template <int K, int PAD, int V, int CF = 0>
void reuse_seven_impl(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  constexpr int P = 7;
  int H = x.size(2), W = x.size(3), OH = v.size(2), OW = v.size(3),
      C = x.size(1), CS = v.stride(3), CT = (C + V - 1) / V,
      XT = (W + 2 * P - 1) / (2 * P), YT = (H + 1) / 2,
      count = x.size(0) * CT * XT * YT,
      span = (x.size(0) * OH * OW - 1) * CS + C;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      EReuseSeven<K, PAD, V, CF>{
          (const uint64_t*)idx.data_ptr(),
          (const sycl::half*)v.data_ptr(),
          (sycl::half*)out.data_ptr(),
          H,
          W,
          OH,
          OW,
          C,
          CS,
          CT,
          XT,
          YT,
          count,
          span,
          EDiv(CT),
          EDiv(XT),
          EDiv(YT)});
}

template <int PAD>
struct IndexByteS2 {
  const uint64_t* src;
  uint32_t* dst;
  int C, HW, W, OW, CT, PT, count;
  EDiv dc, dt, dw;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t t = dc.div(id), cb = (id - t * CT) * 32, n = dt.div(t);
    int p0 = (t - n * PT) * 4;
    e::simd<uint32_t, 128> lo, high;
    if (HW % 2 == 0) {
      auto raw = e::load_2d<uint32_t, 8, 32, 1, true>(
          (const uint32_t*)(src + n * C * HW),
          HW * 8 - 1,
          C - 1,
          HW * 8 - 1,
          p0 * 2,
          cb);
      lo = raw.template bit_cast_view<uint32_t, 8, 32>()
               .template select<4, 2, 32, 1>(0, 0)
               .read();
      high = raw.template bit_cast_view<uint32_t, 8, 32>()
                 .template select<4, 2, 32, 1>(1, 0)
                 .read();
    } else
      paired_index_load(src + n * C * HW, C, HW, cb, p0, lo, high);
    e::simd<uint8_t, 128> encoded;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      uint32_t ph = dw.div(p0 + p), pw = p0 + p - ph * OW;
      int base = (int(ph) * 2 - PAD) * W + int(pw) * 2 - PAD;
      e::simd<uint32_t, 32> delta =
          lo.template select<32, 1>(p * 32) - uint32_t(base);
      e::simd<uint8_t, 32> enc = delta;
      enc.merge(
          uint8_t(255),
          (delta > uint32_t(2 * W + 2)) |
              (high.template select<32, 1>(p * 32) != 0));
      encoded.template select<32, 1>(p * 32) = enc;
    }
    if (C % 16 == 0)
      e::store_2d<uint8_t, 32, 4>(
          (uint8_t*)dst + n * HW * C, C - 1, HW - 1, C - 1, cb, p0, encoded);
    else {
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        if (p0 + p < HW) {
          e::simd<uint8_t, 32> bytes = encoded.template select<32, 1>(p * 32);
          e::simd<uint32_t, 8> v = bytes.template bit_cast_view<uint32_t>();
          auto ptr = dst + ((n * HW + p0 + p) * C + cb) / 4;
          if (cb + 32 <= C)
            e::block_store<uint32_t, 8>(ptr, v, e::properties{e::alignment<4>});
          else {
            e::simd<uint32_t, 8> lane(0, 1);
            e::scatter<uint32_t, 8>(
                ptr, lane * 4, v, lane < uint32_t((C - cb) / 4));
          }
        }
      }
    }
  }
};
template <int PAD, int V>
struct ByteGather2x2 {
  const sycl::vec<sycl::half, V>* go;
  const sycl::vec<uint8_t, V>* idx;
  sycl::vec<sycl::half, V>* gi;
  int H, W, OH, OW, C, CS, XT, YT, count;
  Div dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const {
    constexpr int D = PAD == 0 ? -1 : 0, OVERLAP = PAD == 0 ? 0 : 3;
    for (uint32_t i = it.get_global_linear_id(); i < count;
         i += it.get_global_range(0)) {
      uint32_t tile = dc.div(i), c = i - tile * C;
      uint32_t row = dx.div(tile), xb = tile - row * XT;
      uint32_t n = dy.div(row), yb = row - n * YT;
      sycl::vec<float, V> full(0);
      sycl::vec<sycl::half, V> partial[3];
#pragma unroll
      for (int j = 0; j < 3; ++j)
        partial[j] = sycl::vec<sycl::half, V>(0);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        int ph = int(yb) + D + hh;
        if (ph >= 0 && ph < OH) {
#pragma unroll
          for (int ww = 0; ww < 2; ++ww) {
            int pw = int(xb) + D + ww;
            if (pw >= 0 && pw < OW) {
              auto code = idx[((n * OH + ph) * OW + pw) * C + c];
              auto value = go[((n * OH + ph) * OW + pw) * CS + c];
#pragma unroll
              for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int w = 0; w < 2; ++w) {
                  if ((PAD == 0 ? (hh == 1 || h == 0) : (hh == 0 || h == 1)) &&
                      (PAD == 0 ? (ww == 1 || w == 0) : (ww == 0 || w == 1))) {
                    int target = (h + PAD - 2 * D - 2 * hh) * W + w + PAD -
                        2 * D - 2 * ww;
                    auto match = code == sycl::vec<uint8_t, V>(target);
                    if (h * 2 + w == OVERLAP) {
                      full += sycl::select(
                          sycl::vec<float, V>(0),
                          value.template convert<float>(),
                          match.template convert<int>());
                    } else {
                      int slot =
                          h * 2 + w < OVERLAP ? h * 2 + w : h * 2 + w - 1;
                      partial[slot] += sycl::select(
                          sycl::vec<sycl::half, V>(0),
                          value,
                          match.template convert<short>());
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
#pragma unroll
        for (int w = 0; w < 2; ++w) {
          if (yb * 2 + h < H && xb * 2 + w < W) {
            uint32_t dest = ((n * H + yb * 2 + h) * W + xb * 2 + w) * C + c;
            if (h * 2 + w == OVERLAP)
              gi[dest] =
                  full.template convert<sycl::half, sycl::rounding_mode::rte>();
            else {
              int slot = h * 2 + w < OVERLAP ? h * 2 + w : h * 2 + w - 1;
              gi[dest] = partial[slot];
            }
          }
        }
      }
    }
  }
};

template <int PAD, int V>
void byte_s2_impl(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  int C = x.size(1), H = x.size(2), W = x.size(3), OH = v.size(2),
      OW = v.size(3), HW = OH * OW, CT = (C + 31) / 32, PT = (HW + 3) / 4,
      nidx = x.size(0) * CT * PT;
  auto tmp = at::empty(
      v.sizes(),
      v.options().dtype(at::kByte).memory_format(
          at::MemoryFormat::ChannelsLast));
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(
      sycl::nd_range<1>((nidx + 15) / 16 * 16, 16),
      IndexByteS2<PAD>{
          (const uint64_t*)idx.data_ptr(),
          (uint32_t*)tmp.data_ptr(),
          C,
          HW,
          W,
          OW,
          CT,
          PT,
          nidx,
          EDiv(CT),
          EDiv(PT),
          EDiv(OW)});
  int XT = (W + 1) / 2, YT = (H + 1) / 2, count = x.size(0) * XT * YT * (C / V),
      groups = std::min((count + 255) / 256, 1280);
  q.parallel_for(
      sycl::nd_range<1>(groups * 256, 256),
      ByteGather2x2<PAD, V>{
          (const sycl::vec<sycl::half, V>*)v.data_ptr(),
          (const sycl::vec<uint8_t, V>*)tmp.data_ptr(),
          (sycl::vec<sycl::half, V>*)out.data_ptr(),
          H,
          W,
          OH,
          OW,
          C / V,
          int(v.stride(3) / V),
          XT,
          YT,
          count,
          Div(C / V),
          Div(XT),
          Div(YT)});
}

template <int CF, int H, int IC, int SG>
struct TiledCompactGather {
  const sycl::vec<uint32_t, 4>* go;
  const uint64_t* idx;
  sycl::vec<uint32_t, 4>* gi;
  int CS, count;

  [[sycl::reqd_sub_group_size(SG)]] void operator()(sycl::nd_item<1> it) const {
    constexpr int C = CF / 8, T = (H + 1) / 2;
    for (uint32_t i = it.get_global_linear_id(); i < count;
         i += it.get_global_range(0)) {
      uint32_t tile = i / C, c = i % C;
      uint32_t x = tile % T * 2, y = (tile / T) % T * 2;
      uint32_t n = tile / (T * T);
      sycl::vec<float, 8> acc[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[j] = sycl::vec<float, 8>(0);
#pragma unroll
      for (int h = 0; h < 4; ++h) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          int py = int(y) + h - 1, px = int(x) + w - 1;
          if (py >= 0 && py < H && px >= 0 && px < H) {
            uint32_t pix = (n * H + py) * H + px;
            auto ids = sycl::vec<uint64_t, 1>(idx[pix * (IC / 8) + c])
                           .template as<sycl::vec<uint8_t, 8>>();
            auto val = go[pix * CS + c]
                           .template as<sycl::vec<sycl::half, 8>>()
                           .template convert<float>();
#pragma unroll
            for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
              for (int dx = 0; dx < 2; ++dx) {
                if (h >= dy && h <= dy + 2 && w >= dx && w <= dx + 2) {
                  auto match = ids ==
                      sycl::vec<uint8_t, 8>((dy + 2 - h) * H + dx + 2 - w);
                  acc[dy * 2 + dx] += sycl::select(
                      sycl::vec<float, 8>(0),
                      val,
                      match.template convert<int>());
                }
              }
            }
          }
        }
      }
#pragma unroll
      for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
          if (y + dy < H && x + dx < H) {
            uint32_t dest = ((n * H + y + dy) * H + x + dx) * C + c;
            gi[dest] =
                acc[dy * 2 + dx]
                    .template convert<sycl::half, sycl::rounding_mode::rte>()
                    .template as<sycl::vec<uint32_t, 4>>();
          }
        }
      }
    }
  }
};

template <int C, int H>
struct WideCompactIndex {
  const uint64_t* src;
  uint32_t* dst;
  int tiles, count;
  EDiv dc, dt;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    constexpr int HW = H * H, CT = (C + 63) / 64, CP = CT * 64;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t t = dc.div(id), cb = (id - t * CT) * 64, n = dt.div(t);
    int p0 = (t - n * tiles) * 4;
    e::simd<uint32_t, 128> lo[2], high[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      if constexpr (HW % 2 == 0) {
        auto raw = e::load_2d<uint32_t, 8, 32, 1, true>(
            (const uint32_t*)(src + n * C * HW),
            HW * 8 - 1,
            C - 1,
            HW * 8 - 1,
            p0 * 2,
            cb + c * 32);
        lo[c] = raw.template bit_cast_view<uint32_t, 8, 32>()
                    .template select<4, 2, 32, 1>(0, 0)
                    .read();
        high[c] = raw.template bit_cast_view<uint32_t, 8, 32>()
                      .template select<4, 2, 32, 1>(1, 0)
                      .read();
      } else {
        paired_index_load(
            src + n * C * HW, C, HW, cb + c * 32, p0, lo[c], high[c]);
      }
    }
    e::simd<uint8_t, 256> encoded;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      e::simd<uint32_t, 128> lane(0, 1),
          delta = lo[c] - (lane / 32 + p0) + H + 1;
      e::simd<uint8_t, 128> code = delta;
      code.merge(uint8_t(255), (delta > 2 * H + 2) | (high[c] != 0));
      encoded.template bit_cast_view<uint8_t, 4, 64>()
          .template select<4, 1, 32, 1>(0, c * 32) =
          code.template bit_cast_view<uint8_t, 4, 32>();
    }
    e::store_2d<uint8_t, 64, 4>(
        (uint8_t*)dst + n * HW * CP, CP - 1, HW - 1, CP - 1, cb, p0, encoded);
  }
};

template <int C, int H>
void padded_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  constexpr int CP = (C + 63) / 64 * 64, CT = CP / 64;
  int tiles = (H * H + 3) / 4, nidx = x.size(0) * tiles * CT;
  auto temp = at::empty({x.size(0) * H * H * CP}, v.options().dtype(at::kByte));
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(
      sycl::nd_range<1>((nidx + 15) / 16 * 16, 16),
      WideCompactIndex<C, H>{
          (const uint64_t*)idx.data_ptr(),
          (uint32_t*)temp.data_ptr(),
          tiles,
          nidx,
          EDiv(CT),
          EDiv(tiles)});
  int count = x.size(0) * ((H + 1) / 2) * ((H + 1) / 2) * (C / 8),
      groups = std::min((count + 255) / 256, 1280);
  q.parallel_for(
      sycl::nd_range<1>(groups * 256, 256),
      TiledCompactGather<C, H, CP, (C == 216 ? 32 : 16)>{
          (const sycl::vec<uint32_t, 4>*)v.data_ptr(),
          (const uint64_t*)temp.data_ptr(),
          (sycl::vec<uint32_t, 4>*)out.data_ptr(),
          int(v.stride(3) / 8),
          count});
}

template <int C, int P, int V>
struct PackedForwardS1 {
  const sycl::half* x;
  sycl::half* y;
  uint64_t* idx;
  int H, W, pixels, count;
  EDiv dw, dh;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t base = id * P, nh = dw.div(base), ox = base - nh * W;
    uint32_t n = dh.div(nh), oy = nh - n * H;
    e::simd<sycl::half, V> best[P];
    e::simd<uint16_t, V> bi[P];
    e::simd<uint32_t, V> lane(0, 1);
#pragma unroll
    for (int p = 0; p < P; ++p) {
      best[p] = sycl::half(-INFINITY);
      uint32_t row = dw.div(base + p), px = base + p - row * W;
      uint32_t nn = dh.div(row), py = row - nn * H;
      bi[p] = uint16_t((py > 0 ? py - 1 : 0) * W + (px > 0 ? px - 1 : 0));
    }
    if (ox + P <= W && base + P <= pixels) {
#pragma unroll
      for (int h = 0; h < 3; ++h) {
        int py = int(oy) + h - 1;
        if (py >= 0 && py < H) {
#pragma unroll
          for (int w = 0; w < P + 2; ++w) {
            int px = int(ox) + w - 1;
            if (px >= 0 && px < W) {
              auto ptr = x + ((n * H + py) * W + px) * C;
              e::simd<sycl::half, V> value;
              if (((n * H + py) * W + px) * C + V <= uint32_t(pixels * C))
                value = e::block_load<sycl::half, V>(
                    ptr, e::properties{e::alignment<4>});
              else
                value = e::gather<sycl::half, V>(ptr, lane * 2, lane < C);
#pragma unroll
              for (int p = 0; p < P; ++p) {
                if (w >= p && w <= p + 2) {
                  auto take = (value > best[p]) | (value != value);
                  best[p].merge(value, take);
                  bi[p].merge(uint16_t(py * W + px), take);
                }
              }
            }
          }
        }
      }
    } else {
#pragma unroll
      for (int p = 0; p < P; ++p) {
        if (base + p < pixels) {
          uint32_t row = dw.div(base + p), px0 = base + p - row * W;
          uint32_t nn = dh.div(row), py0 = row - nn * H;
#pragma unroll
          for (int h = 0; h < 3; ++h) {
#pragma unroll
            for (int w = 0; w < 3; ++w) {
              int py = int(py0) + h - 1, px = int(px0) + w - 1;
              if (py >= 0 && py < H && px >= 0 && px < W) {
                auto ptr = x + ((nn * H + py) * W + px) * C;
                auto value = e::gather<sycl::half, V>(ptr, lane * 2, lane < C);
                auto take = (value > best[p]) | (value != value);
                best[p].merge(value, take);
                bi[p].merge(uint16_t(py * W + px), take);
              }
            }
          }
        }
      }
    }
    e::simd<uint16_t, C * P> indices;
#pragma unroll
    for (int p = 0; p < P; ++p)
      indices.template select<C, 1>(p * C) = bi[p].template select<C, 1>(0);
    uint32_t valid = (pixels - base < P ? pixels - base : P) * C;
    e::simd<uint32_t, 32> ilane(0, 1);
#pragma unroll
    for (int j = 0; j < C * P; j += 32) {
      e::simd<uint64_t, 32> value = indices.template select<32, 1>(j);
      if (j >= valid)
        continue;
      auto ptr = idx + uint64_t(base) * C + j;
      if (j + 32 <= valid)
        e::block_store<uint64_t, 32>(
            ptr, value, e::properties{e::alignment<16>});
      else
        e::scatter<uint64_t, 32>(ptr, ilane * 8, value, ilane + j < valid);
    }
    e::simd<sycl::half, C * P> values;
#pragma unroll
    for (int p = 0; p < P; ++p)
      values.template select<C, 1>(p * C) = best[p].template select<C, 1>(0);
    e::simd<uint32_t, C * P / 2> words =
        values.template bit_cast_view<uint32_t>();
    e::simd<uint32_t, 16> vlane(0, 1);
#pragma unroll
    for (int j = 0; j < C * P / 2; j += 16) {
      auto value = words.template select<16, 1>(j).read();
      if (j >= valid / 2)
        continue;
      auto ptr = (uint32_t*)(y + base * C) + j;
      if (j + 16 <= valid / 2)
        e::block_store<uint32_t, 16>(
            ptr, value, e::properties{e::alignment<4>});
      else
        e::scatter<uint32_t, 16>(ptr, vlane * 4, value, vlane + j < valid / 2);
    }
  }
};

template <int C, int P, int V>
void packed_forward_launch(
    const at::Tensor& x,
    const at::Tensor& y,
    const at::Tensor& idx) {
  int pixels = x.numel() / C, count = (pixels + P - 1) / P;
  int H = x.size(2), W = x.size(3);
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      PackedForwardS1<C, P, V>{
          (const sycl::half*)x.data_ptr(),
          (sycl::half*)y.data_ptr(),
          (uint64_t*)idx.data_ptr(),
          H,
          W,
          pixels,
          count,
          EDiv(W),
          EDiv(H)});
}

template <int P, int CF = 0>
struct EBackwardSeven {
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<128>};
  }
  const uint64_t* idx;
  const sycl::half* go;
  sycl::half* gi;
  int H, W, OH, OW, C, CS, CT, XT, YT, count;
  EDiv dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = dc.div(id), c = (id - q * CT) * 32, r = dx.div(q),
             px = (q - r * XT) * 7, n = dy.div(r), yb = r - n * YT;
    int HW = OH * OW;
    constexpr int D = P == 0 ? -1 : 0;
    e::simd<float, 224> acc_overlap = 0;
    e::simd<sycl::half, 224> acc_half[3];
#pragma unroll
    for (int a = 0; a < 3; ++a)
      acc_half[a] = sycl::half(0);
    e::simd<uint32_t, 224> lane(0, 1);
    e::simd<uint16_t, 224> target = yb * 2 * W + (px + lane / 32) * 2;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      int ph = int(yb) + D + hh;
      if (ph >= 0 && ph < OH) {
        int pw = int(px) + D, pbase = ph * OW + pw;
        e::simd<uint16_t, 256> indices;
        e::simd<sycl::half, 256> half;
#pragma unroll
        for (int block = 0; block < 2; ++block) {
          e::simd<uint32_t, 128> lo, high;
          int pos = pbase + block * 4;
          if (HW % 2 == 0) {
            auto raw = e::load_2d<uint32_t, 8, 32, 1, true>(
                (const uint32_t*)(idx + n * C * HW),
                HW * 8 - 1,
                C - 1,
                HW * 8 - 1,
                pos * 2,
                c);
            lo = raw.template bit_cast_view<uint32_t, 8, 32>()
                     .template select<4, 2, 32, 1>(0, 0)
                     .read();
            high = raw.template bit_cast_view<uint32_t, 8, 32>()
                       .template select<4, 2, 32, 1>(1, 0)
                       .read();
          } else
            paired_index_load(idx + n * C * HW, C, HW, c, pos, lo, high);
          lo.merge(uint32_t(65535), (high != 0) | (lo >= 65535));
          indices.template select<128, 1>(block * 128) = lo;
          if (CS % 8 == 0)
            half.template select<128, 1>(block * 128) =
                e::load_2d<sycl::half, 32, 4>(
                    go + n * HW * CS, C * 2 - 1, HW - 1, CS * 2 - 1, c, pos);
          else {
#pragma unroll
            for (int parity = 0; parity < 2; ++parity) {
              int pp = pos + parity;
              auto pair = e::load_2d<sycl::half, 32, 2>(
                  go + n * HW * CS,
                  (CS + C) * 2 - 1,
                  HW / 2 - 1,
                  CS * 4 - 1,
                  (pp & 1) * CS + c,
                  pp >> 1);
              half.template bit_cast_view<sycl::half, 8, 32>()
                  .template select<2, 2, 32, 1>(block * 4 + parity, 0) = pair;
            }
          }
        }

#pragma unroll
        for (int ww = 0; ww < 2; ++ww) {
          e::simd<uint16_t, 224> ix = indices.template select<224, 1>(ww * 32);
          e::simd<sycl::half, 224> v = half.template select<224, 1>(ww * 32);
          auto valid = (lane / 32 + pw + ww < uint32_t(OW));
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int w = 0; w < 2; ++w) {
              if ((P == 1 ? (hh == 0 || h == 1) : (hh == 1 || h == 0)) &&
                  (P == 1 ? (ww == 0 || w == 1) : (ww == 1 || w == 0))) {
                e::simd<sycl::half, 224> add = v;
                add.merge(sycl::half(0), (ix != target + h * W + w) | !valid);
                if (h * 2 + w == (P == 1 ? 3 : 0)) {
                  e::simd<float, 224> value = add;
                  acc_overlap += value;
                } else
                  acc_half[(P == 1 ? h * 2 + w : h * 2 + w - 1)] += add;
              }
            }
          }
        }
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      if (yb * 2 + h < H) {
        if constexpr (CF > 0) {
          constexpr int WORDS = 7 * CF;
          e::simd<sycl::half, WORDS * 2> row;
#pragma unroll
          for (int pp = 0; pp < 7; ++pp) {
#pragma unroll
            for (int w = 0; w < 2; ++w) {
              e::simd<sycl::half, CF> v;
              if (h * 2 + w == (P == 1 ? 3 : 0))
                v = acc_overlap.template select<CF, 1>(pp * 32);
              else
                v = acc_half[(P == 1 ? h * 2 + w : h * 2 + w - 1)]
                        .template select<CF, 1>(pp * 32);
              row.template select<CF, 1>((pp * 2 + w) * CF) = v;
            }
          }
          e::simd<uint32_t, WORDS> packed =
              row.template bit_cast_view<uint32_t>();
          int valid = (W - int(px * 2) < 14 ? W - int(px * 2) : 14) * (CF / 2);
          auto dst = (uint32_t*)(gi + ((n * H + yb * 2 + h) * W + px * 2) * CF);
#pragma unroll
          for (int j = 0; j < WORDS / 128 * 128; j += 128) {
            if (j < valid) {
              auto v = packed.template select<128, 1>(j).read();
              if (j + 128 <= valid)
                e::block_store<uint32_t, 128>(
                    dst + j, v, e::properties{e::alignment<4>});
              else {
                e::simd<uint32_t, 128> l(0, 1);
                e::scatter<uint32_t, 128>(
                    dst + j, l * 4, v, l + j < uint32_t(valid));
              }
            }
          }
#pragma unroll
          for (int j = WORDS / 128 * 128; j < WORDS / 32 * 32; j += 32) {
            if (j < valid) {
              auto v = packed.template select<32, 1>(j).read();
              if (j + 32 <= valid)
                e::block_store<uint32_t, 32>(
                    dst + j, v, e::properties{e::alignment<4>});
              else {
                e::simd<uint32_t, 32> l(0, 1);
                e::scatter<uint32_t, 32>(
                    dst + j, l * 4, v, l + j < uint32_t(valid));
              }
            }
          }
#pragma unroll
          for (int j = WORDS / 32 * 32; j < WORDS; j += 8) {
            if (j < valid) {
              auto v = packed.template select<8, 1>(j).read();
              if (j + 8 <= valid)
                e::block_store<uint32_t, 8>(
                    dst + j, v, e::properties{e::alignment<4>});
              else {
                e::simd<uint32_t, 8> l(0, 1);
                e::scatter<uint32_t, 8>(
                    dst + j, l * 4, v, l + j < uint32_t(valid));
              }
            }
          }
        } else {
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            e::simd<sycl::half, 224> v;
            if (h * 2 + w == (P == 1 ? 3 : 0))
              v = acc_overlap;
            else
              v = acc_half[(P == 1 ? h * 2 + w : h * 2 + w - 1)];
            e::store_2d<sycl::half, 32, 7>(
                gi + ((n * H + yb * 2 + h) * W + w) * C,
                C * 2 - 1,
                (W + 1 - w) / 2 - 1,
                C * 4 - 1,
                c,
                px,
                v);
          }
        }
      }
    }
  }
};
template <int P, int CF = 0>
void eb7_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  int H = x.size(2), W = x.size(3), OH = v.size(2), OW = v.size(3),
      C = x.size(1), CT = (C + 31) / 32, XT = (W + 13) / 14, YT = (H + 1) / 2,
      count = x.size(0) * YT * XT * CT;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      EBackwardSeven<P, CF>{
          (const uint64_t*)idx.data_ptr(),
          (const sycl::half*)v.data_ptr(),
          (sycl::half*)out.data_ptr(),
          H,
          W,
          OH,
          OW,
          C,
          int(v.stride(3)),
          CT,
          XT,
          YT,
          count,
          EDiv(CT),
          EDiv(XT),
          EDiv(YT)});
}

template <int K, int PAD, int V, int CF = 0>
struct EReuse108 {
  const uint64_t* idx;
  const sycl::half* go;
  sycl::half* gi;
  int H, W, OH, OW, C, CS, CT, XT, YT, count, span;
  EDiv dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    constexpr int P = 3, NH = K == 2 ? 1 : 2, SW = 4, NG = P + NH - 1,
                  D = (K == 3 && PAD == 0) ? -1 : 0;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = dc.div(id), cb = (id - q * CT) * V, r = dx.div(q),
             px = (q - r * XT) * P, n = dy.div(r), yb = r - n * YT;
    int HW = OH * OW;
    e::simd<float, 384> acc_overlap = 0;
    e::simd<sycl::half, 384> acc_half[3];
#pragma unroll
    for (int j = 0; j < 3; ++j)
      acc_half[j] = sycl::half(0);
    e::simd<uint32_t, 64> lane(0, 1);
#pragma unroll
    for (int hh = 0; hh < NH; ++hh) {
      int ph = int(yb) + D + hh;
      if (ph >= 0 && ph < OH) {
        e::simd<uint16_t, V * SW> indices;
#pragma unroll
        for (int block = 0; block < 1; ++block) {
#pragma unroll
          for (int cb = 0; cb < 128; cb += 32) {
            e::simd<uint32_t, 128> lo, high;
            int pos = ph * OW + px + D + block * 4;
            if (HW % 2 == 0) {
              auto raw = e::load_2d<uint32_t, 8, 32, 1, true>(
                  (const uint32_t*)(idx + n * C * HW),
                  HW * 8 - 1,
                  C - 1,
                  HW * 8 - 1,
                  pos * 2,
                  cb);
              lo = raw.template bit_cast_view<uint32_t, 8, 32>()
                       .template select<4, 2, 32, 1>(0, 0)
                       .read();
              high = raw.template bit_cast_view<uint32_t, 8, 32>()
                         .template select<4, 2, 32, 1>(1, 0)
                         .read();
            } else
              paired_index_load(idx + n * C * HW, C, HW, cb, pos, lo, high);
            e::simd<uint16_t, 128> compact = lo;
            compact.merge(uint16_t(65535), (lo >= 65535) | (high != 0));
#pragma unroll
            for (int pp = 0; pp < 4; ++pp)
              indices.template select<32, 1>((block * 4 + pp) * 128 + cb) =
                  compact.template select<32, 1>(pp * 32);
          }
        }
        e::simd<sycl::half, V> values[NG];
#pragma unroll
        for (int j = 0; j < NG; ++j) {
          int pw = int(px) + D + j;
          values[j] = 0;
          if (pw >= 0 && pw < OW) {
            int off = ((n * OH + ph) * OW + pw) * CS + cb;
            if (off + V <= span)
              values[j] = e::block_load<sycl::half, V>(
                  go + off, e::properties{e::alignment<4>});
            else {
              e::simd<uint32_t, V> c(0, 1);
              values[j] = e::gather<sycl::half, V>(
                  go + off, c * 2, c < uint32_t(C - cb));
            }
          }
        }
#pragma unroll
        for (int p = 0; p < P; ++p) {
#pragma unroll
          for (int ww = 0; ww < NH; ++ww) {
            e::simd<uint16_t, V> ix =
                indices.template select<V, 1>((p + ww) * V);
            e::simd<sycl::half, V> v = values[p + ww];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
              for (int w = 0; w < 2; ++w) {
                if (K == 2 ||
                    ((PAD == 1 ? (hh == 0 || h == 1) : (hh == 1 || h == 0)) &&
                     (PAD == 1 ? (ww == 0 || w == 1) : (ww == 1 || w == 0)))) {
                  e::simd<sycl::half, V> add = v;
                  add.merge(
                      sycl::half(0),
                      ix != uint16_t((yb * 2 + h) * W + (px + p) * 2 + w));
                  if (h * 2 + w == (PAD == 1 ? 3 : 0)) {
                    e::simd<float, V> val = add;
                    acc_overlap.template select<V, 1>(p * V) += val;
                  } else
                    acc_half[(PAD == 1 ? h * 2 + w : h * 2 + w - 1)]
                        .template select<V, 1>(p * V) += add;
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
        if constexpr (CF > 0) {
          constexpr int WORDS = P * CF, PW = (WORDS + 15) / 16 * 16;
          e::simd<sycl::half, PW * 2> row = 0;
#pragma unroll
          for (int p = 0; p < P; ++p) {
#pragma unroll
            for (int w = 0; w < 2; ++w) {
              e::simd<sycl::half, CF> val;
              if (h * 2 + w == (PAD == 1 ? 3 : 0))
                val = acc_overlap.template select<CF, 1>(p * V);
              else
                val = acc_half[(PAD == 1 ? h * 2 + w : h * 2 + w - 1)]
                          .template select<CF, 1>(p * V);
              row.template select<CF, 1>(p * CF * 2 + w * CF) = val;
            }
          }
          e::simd<uint32_t, PW> packed = row.template bit_cast_view<uint32_t>();
          auto target =
              (uint32_t*)(gi + ((n * H + yb * 2 + h) * W + px * 2) * C);
          int valid =
              (int(P * 2) < W - int(px * 2) ? int(P * 2) : W - int(px * 2)) *
              CF / 2;
#pragma unroll
          for (int j = 0; j < WORDS / 128; ++j) {
            auto v = packed.template select<128, 1>(j * 128).read();
            if (valid >= j * 128 + 128)
              e::block_store<uint32_t, 128>(
                  target + j * 128, v, e::properties{e::alignment<4>});
            else {
              e::simd<uint32_t, 128> l(0, 1);
              e::scatter<uint32_t, 128>(
                  target + j * 128, l * 4, v, l + j * 128 < uint32_t(valid));
            }
          }
#pragma unroll
          for (int j = WORDS / 128 * 128; j < PW; j += 16) {
            auto v = packed.template select<16, 1>(j).read();
            if (valid >= j + 16)
              e::block_store<uint32_t, 16>(
                  target + j, v, e::properties{e::alignment<4>});
            else {
              e::simd<uint32_t, 16> l(0, 1);
              e::scatter<uint32_t, 16>(
                  target + j, l * 4, v, l + j < uint32_t(valid));
            }
          }
        } else {
#pragma unroll
          for (int p = 0; p < P; ++p) {
#pragma unroll
            for (int w = 0; w < 2; ++w)
              if ((px + p) * 2 + w < W) {
                e::simd<sycl::half, V> v;
                if (h * 2 + w == (PAD == 1 ? 3 : 0))
                  v = acc_overlap.template select<V, 1>(p * V);
                else
                  v = acc_half[(PAD == 1 ? h * 2 + w : h * 2 + w - 1)]
                          .template select<V, 1>(p * V);
                e::block_store<sycl::half, V>(
                    gi + ((n * H + yb * 2 + h) * W + (px + p) * 2 + w) * C + cb,
                    v,
                    e::properties{e::alignment<16>});
              }
          }
        }
      }
    }
  }
};
template <int K, int PAD, int V, int CF = 0>
void reuse108_impl(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  constexpr int P = 3;
  int H = x.size(2), W = x.size(3), OH = v.size(2), OW = v.size(3),
      C = x.size(1), CS = v.stride(3), CT = (C + V - 1) / V,
      XT = (W + 2 * P - 1) / (2 * P), YT = (H + 1) / 2,
      count = x.size(0) * CT * XT * YT,
      span = (x.size(0) * OH * OW - 1) * CS + C;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      EReuse108<K, PAD, V, CF>{
          (const uint64_t*)idx.data_ptr(),
          (const sycl::half*)v.data_ptr(),
          (sycl::half*)out.data_ptr(),
          H,
          W,
          OH,
          OW,
          C,
          CS,
          CT,
          XT,
          YT,
          count,
          span,
          EDiv(CT),
          EDiv(XT),
          EDiv(YT)});
}

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
template <int K, int P, bool Partial = false>
struct EForwardThree {
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<128>};
  }
  const sycl::half* x;
  sycl::half* y;
  uint64_t* idx;
  int H, W, OH, OW, C, CT, XT, count, NP;
  EDiv dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = dc.div(id), c = (id - q * CT) * 32, r = dx.div(q),
             ox = (q - r * XT) * 3, n = dy.div(r), oy = r - n * OH;
    int h0 = int(oy) * 2 - P, w0 = int(ox) * 2 - P;
    e::simd<uint32_t, 96> lane(0, 1);
    e::simd<int32_t, 96> left = w0 + e::convert<int32_t>(lane / 32) * 2;
    auto first = left;
    first.merge(0, first < 0);
    e::simd<uint32_t, 96> bi = (h0 > 0 ? h0 : 0) * W + first;
    e::simd<sycl::half, 96> best = sycl::half(-INFINITY);
#pragma unroll
    for (int h = 0; h < K; ++h) {
      int ih = h0 + h;
      if (ih >= 0 && ih < H) {
        e::simd<sycl::half, 32 * (4 + K)> raw;
        if constexpr (Partial)
          raw = e::load_2d<sycl::half, 32, 4 + K>(
              x, C * 2 - 1, NP - 1, C * 2 - 1, c, (n * H + ih) * W + w0);
        else
          raw = e::load_2d<sycl::half, 32, 4 + K>(
              x + (n * H + ih) * W * C, C * 2 - 1, W - 1, C * 2 - 1, c, w0);
#pragma unroll
        for (int w = 0; w < K; ++w) {
          e::simd<sycl::half, 96> v =
              raw.template bit_cast_view<sycl::half, 4 + K, 32>()
                  .template select<3, 2, 32, 1>(w, 0)
                  .read();
          auto col = left + w;
          v.merge(sycl::half(-INFINITY), (col < 0) | (col >= W));
          auto take = (v > best) | (v != v);
          best.merge(v, take);
          bi.merge(e::convert<uint32_t>(ih * W + col), take);
        }
      }
    }
#pragma unroll
    for (int pp = 0; pp < 3; ++pp) {
      if (ox + pp < OW) {
        uint32_t dest = ((n * OH + oy) * OW + ox + pp) * C + c;
        e::simd<sycl::half, 32> v = best.template select<32, 1>(pp * 32);
        e::simd<uint64_t, 32> ix = bi.template select<32, 1>(pp * 32);
        if (!Partial || c + 32 <= C) {
          e::block_store<sycl::half, 32>(
              y + dest, v, e::properties{e::alignment<4>});
          e::block_store<uint64_t, 32>(
              idx + uint64_t(dest), ix, e::properties{e::alignment<16>});
        } else {
          e::simd<sycl::half, 16> tail = v.template select<16, 1>(0);
          e::simd<uint64_t, 16> tail_index = ix.template select<16, 1>(0);
          e::block_store<sycl::half, 16>(
              y + dest, tail, e::properties{e::alignment<4>});
          e::block_store<uint64_t, 16>(
              idx + uint64_t(dest),
              tail_index,
              e::properties{e::alignment<16>});
        }
      }
    }
  }
};
template <int K, int P, bool Partial = false>
void ef_three_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx) {
  int C = x.size(1), CT = (C + 31) / 32, XT = (v.size(3) + 2) / 3,
      count = x.size(0) * v.size(2) * XT * CT;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      EForwardThree<K, P, Partial>{
          (const sycl::half*)x.data_ptr(),
          (sycl::half*)v.data_ptr(),
          (uint64_t*)idx.data_ptr(),
          int(x.size(2)),
          int(x.size(3)),
          int(v.size(2)),
          int(v.size(3)),
          C,
          CT,
          XT,
          count,
          int(x.size(0) * x.size(2) * x.size(3)),
          EDiv(CT),
          EDiv(XT),
          EDiv(v.size(2))});
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
  if (reinterpret_cast<uintptr_t>(x.data_ptr()) % 4 ||
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
  if (!bmg_device())
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
  if (S == 2 && K == 3 && x.size(1) == 432 &&
      x.size(0) * x.size(2) * x.size(3) < (1 << 24) &&
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 64 == 0) {
    if (P == 0)
      ef_three_launch<3, 0, true>(x, v, idx);
    else
      ef_three_launch<3, 1, true>(x, v, idx);
  } else if (
      S == 2 && K >= 2 && x.size(1) % 32 == 0 && x.size(1) <= (1 << 23) &&
      x.size(3) < (1 << 24) &&
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 64 == 0) {
    if (K == 2)
      ef_three_launch<2, 0>(x, v, idx);
    else if (P == 0)
      ef_three_launch<3, 0>(x, v, idx);
    else
      ef_three_launch<3, 1>(x, v, idx);
  } else if (S == 2 && x.size(1) % 64 == 0 && K >= 2) {
    if (K == 2)
      ef_launch<2, 2, 0>(x, v, idx);
    else if (P == 0)
      ef_launch<3, 2, 0>(x, v, idx);
    else
      ef_launch<3, 2, 1>(x, v, idx);
  } else if (K >= 5) {
    if (K == 5)
      large_launch<5>(x, v, idx);
    else if (K == 9)
      large_launch<9>(x, v, idx);
    else
      large_launch<13>(x, v, idx);
  } else if (K == 3 && S == 1) {
    if (x.size(1) == 216)
      packed_forward_launch<216, 4, 256>(x, v, idx);
    else
      eforward_s1_launch(x, v, idx);
  } else if (K == 2)
    forward_launch<2, 2, 0>(x, v, idx);
  else if (K == 1)
    forward_launch<1, 2, 0>(x, v, idx);
  else if (P == 0)
    forward_launch<3, 2, 0>(x, v, idx);
  else
    forward_launch<3, 2, 1>(x, v, idx);
  return true;
}
template <int W, int V>
struct WholeRowBackward {
  const uint64_t* idx;
  const sycl::half* go;
  sycl::half* gi;
  int H, OH, C, CS, CT, YT, count;
  EDiv dc, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    constexpr int OW = (W - 1) / 2, T = (W + 1) / 2, R = (OW + 3) / 4 * 4;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t row = dc.div(id), cb = (id - row * CT) * V;
    uint32_t n = dy.div(row), yb = row - n * YT;
    int HW = OH * OW;
    e::simd<float, T * V> full = 0;
    e::simd<sycl::half, T * V> partial[3];
#pragma unroll
    for (int j = 0; j < 3; ++j)
      partial[j] = sycl::half(0);
    e::simd<uint32_t, T * V> lane(0, 1);
    e::simd<uint16_t, T * V> target = yb * 2 * W + lane / V * 2;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      int ph = int(yb) - 1 + hh;
      if (ph >= 0 && ph < OH) {
        e::simd<uint16_t, (R + 2) * V> codes = uint16_t(65535);
        e::simd<sycl::half, (R + 2) * V> values = sycl::half(0);
#pragma unroll
        for (int block = 0; block < R / 4; ++block) {
          int pos = ph * OW + block * 4;
          e::simd<uint32_t, 4 * V> low, high;
          if (HW % 2 == 0) {
            auto raw = e::load_2d<uint32_t, 8, V, 1, true>(
                (const uint32_t*)(idx + n * C * HW),
                HW * 8 - 1,
                C - 1,
                HW * 8 - 1,
                pos * 2,
                cb);
            low = raw.template bit_cast_view<uint32_t, 8, V>()
                      .template select<4, 2, V, 1>(0, 0)
                      .read();
            high = raw.template bit_cast_view<uint32_t, 8, V>()
                       .template select<4, 2, V, 1>(1, 0)
                       .read();
          } else {
#pragma unroll
            for (int parity = 0; parity < 2; ++parity) {
              auto raw = e::load_2d<uint32_t, 8, V / 2, 1, true>(
                  (const uint32_t*)(idx + n * C * HW),
                  HW * 16 - 1,
                  C / 2 - 1,
                  HW * 16 - 1,
                  (pos + parity * HW) * 2,
                  cb / 2);
              low.template bit_cast_view<uint32_t, 4, V>()
                  .template select<4, 1, V / 2, 2>(0, parity) =
                  raw.template bit_cast_view<uint32_t, 8, V / 2>()
                      .template select<4, 2, V / 2, 1>(0, 0);
              high.template bit_cast_view<uint32_t, 4, V>()
                  .template select<4, 1, V / 2, 2>(0, parity) =
                  raw.template bit_cast_view<uint32_t, 8, V / 2>()
                      .template select<4, 2, V / 2, 1>(1, 0);
            }
          }
          e::simd<uint32_t, 4 * V> ilane(0, 1);
          low.merge(
              uint32_t(65535),
              (high != 0) | (low >= 65535) | (ilane / V + block * 4 >= OW));
          codes.template select<4 * V, 1>((block * 4 + 1) * V) = low;
          values.template select<4 * V, 1>((block * 4 + 1) * V) =
              e::load_2d<sycl::half, V, 4>(
                  go + n * HW * CS, C * 2 - 1, HW - 1, CS * 2 - 1, cb, pos);
        }
#pragma unroll
        for (int ww = 0; ww < 2; ++ww) {
          e::simd<uint16_t, T * V> code =
              codes.template select<T * V, 1>(ww * V);
          e::simd<sycl::half, T * V> value =
              values.template select<T * V, 1>(ww * V);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int w = 0; w < 2; ++w) {
              if ((hh == 1 || h == 0) && (ww == 1 || w == 0)) {
                auto add = value;
                add.merge(sycl::half(0), code != target + h * W + w);
                if (h * 2 + w == 0)
                  full += e::convert<float>(add);
                else
                  partial[h * 2 + w - 1] += add;
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
          e::simd<sycl::half, T * V> value;
          if (h * 2 + w == 0)
            value = full;
          else
            value = partial[h * 2 + w - 1];
#pragma unroll
          for (int block = 0; block < T / 8; ++block) {
            e::simd<sycl::half, V * 8> part =
                value.template select<V * 8, 1>(block * V * 8);
            e::store_2d<sycl::half, V, 8>(
                gi + ((n * H + yb * 2 + h) * W + w) * C,
                C * 2 - 1,
                (W + 1 - w) / 2 - 1,
                C * 4 - 1,
                cb,
                block * 8,
                part);
          }
          if constexpr (T % 8 != 0) {
            e::simd<sycl::half, V*(T % 8)> part =
                value.template select<V*(T % 8), 1>(T / 8 * V * 8);
            e::store_2d<sycl::half, V, T % 8>(
                gi + ((n * H + yb * 2 + h) * W + w) * C,
                C * 2 - 1,
                (W + 1 - w) / 2 - 1,
                C * 4 - 1,
                cb,
                T / 8 * 8,
                part);
          }
        }
      }
    }
  }
};
template <int W, int V>
void whole_row_launch(
    const at::Tensor& x,
    const at::Tensor& v,
    const at::Tensor& idx,
    const at::Tensor& out) {
  int H = x.size(2), C = x.size(1), CT = (C + V - 1) / V, YT = (H + 1) / 2,
      count = x.size(0) * YT * CT;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      WholeRowBackward<W, V>{
          (const uint64_t*)idx.data_ptr(),
          (const sycl::half*)v.data_ptr(),
          (sycl::half*)out.data_ptr(),
          H,
          int(v.size(2)),
          C,
          int(v.stride(3)),
          CT,
          YT,
          count,
          EDiv(CT),
          EDiv(YT)});
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
      x.scalar_type() != at::kHalf || v.scalar_type() != at::kHalf ||
      out.scalar_type() != at::kHalf || idx.scalar_type() != at::kLong ||
      !x.numel() || !v.numel() || x.numel() > INT_MAX || v.numel() > INT_MAX ||
      !parameters(k, s, p, d, K, S, P) || K > 3 || K < 2)
    return false;
  int64_t C = x.size(1), H = x.size(2), W = x.size(3), OH = v.size(2),
          OW = v.size(3), CS = v.stride(3);
  if (OH * OW > (1 << 20) || CS > (1 << 23))
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
  if (!bmg_device())
    return false;
  if (S == 1) {
    if (C == 864)
      padded_launch<864, 11>(x, v, idx, out);
    else if (C == 432)
      padded_launch<432, 21>(x, v, idx, out);
    else
      padded_launch<216, 42>(x, v, idx, out);
  } else if (C == 54 && K == 3 && P == 0 && H * W < 65535)
    reuse_seven_impl<3, 0, 64, 54>(x, v, idx, out);
  else if (C == 108 && K == 3 && P == 0 && H * W < 65535)
    reuse108_impl<3, 0, 128, 108>(x, v, idx, out);
  else if (K == 2 && C % 32 == 0 && CS % 8 == 0 && H % 2 == 0 && W % 2 == 0)
    esimd2_launch(x, v, idx, out);
  else if (
      K == 3 && P == 0 && W == 17 && OW == 8 && C >= 128 && C % 32 == 0 &&
      CS % 8 == 0 && H * W < 65535 && out.numel() >= (10LL << 20))
    whole_row_launch<17, 32>(x, v, idx, out);
  else if (
      K == 3 && P == 0 && C >= 128 && W <= 43 && out.numel() >= (10LL << 20) &&
      C % 8 == 0 && CS % 8 == 0 && ((W + 13) / 14) * 14 * 4 > W * 5)
    byte_s2_impl<0, 8>(x, v, idx, out);
  else if (K == 3 && C == 24 && CS % 8 == 0 && H * W < 65535) {
    if (P == 0)
      eb7_launch<0, 24>(x, v, idx, out);
    else
      eb7_launch<1, 24>(x, v, idx, out);
  } else if (
      K == 3 && C % 4 == 0 && C != 432 && CS % 4 == 0 &&
      (CS % 8 == 0 || (OH * OW) % 2 == 0)) {
    if (H * W < 65535 && CS % 8 == 0) {
      if (P == 0)
        eb7_launch<0>(x, v, idx, out);
      else
        eb7_launch<1>(x, v, idx, out);
    } else if (W > 14 && W <= 32 && C % 32 == 0 && CS % 8 == 0) {
      if (P == 0)
        eb3t_launch<0>(x, v, idx, out);
      else
        eb3t_launch<1>(x, v, idx, out);
    } else {
      if (P == 0)
        eb3_launch<0>(x, v, idx, out);
      else
        eb3_launch<1>(x, v, idx, out);
    }
  } else if (C % 4 == 0 && CS % 4 == 0) {
    if (K == 2)
      block4_launch<2, 0>(x, v, idx, out);
    else if (P == 0)
      block4_launch<3, 0>(x, v, idx, out);
    else
      block4_launch<3, 1>(x, v, idx, out);
  } else {
    if (K == 2)
      block_launch<2, 0>(x, v, idx, out);
    else if (P == 0)
      block_launch<3, 0>(x, v, idx, out);
    else
      block_launch<3, 1>(x, v, idx, out);
  }
  return true;
}

} // namespace at::native::xpu::bmg_maxpool
