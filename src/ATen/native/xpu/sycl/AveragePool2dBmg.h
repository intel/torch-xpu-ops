/*
 * Copyright 2020-2026 Intel Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <climits>
namespace at::native::xpu::bmg_avg_pool {
struct Div {
  uint32_t d, m;
  Div(uint32_t a) : d(a), m(a == 1 ? 0 : uint32_t((uint64_t(1) << 32) / a)) {}
  uint32_t div(uint32_t x) const {
    if (d == 1)
      return x;
    uint32_t q = sycl::mul_hi(x, m);
    return q + uint32_t(x - q * d >= d);
  }
};
template <int V>
void store(sycl::half* p, sycl::vec<sycl::half, V> v) {
  v.template as<sycl::vec<uint32_t, V / 2>>().store(
      0, reinterpret_cast<uint32_t*>(p));
}
template <int K, int S, int P, bool Include, int V>
struct Tile {
  const sycl::half* src;
  sycl::half* dst;
  int C, H, W, OH, OW, CS, tx, ty, count;
  Div dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const {
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int q = dc.div(id), c = (id - q * dc.d) * V;
    int r = dx.div(q), x = (q - r * dx.d) * 2;
    int n = dy.div(r), y = (r - n * dy.d) * 2;
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
          sycl::vec<sycl::half, V> v;
          v.load(0, src + ((n * OH + sy) * OW + sx) * CS + c);
          float terms[V];
#pragma unroll
          for (int j = 0; j < V; ++j)
            terms[j] = float(sycl::half(float(v[j]) / divisor));
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
          sycl::vec<sycl::half, V> v;
#pragma unroll
          for (int j = 0; j < V; ++j)
            v[j] = sycl::half(acc[yy * 2 + xx][j]);
          store<V>(dst + ((n * H + y + yy) * W + x + xx) * C + c, v);
        }
      }
    }
  }
};

template <int K, int S, int P, bool Include, int V>
struct Linear {
  const sycl::half* src;
  sycl::half* dst;
  int C, H, W, OH, OW, CS, count;
  Div dc, dw, dh;
  void operator()(sycl::nd_item<1> it) const {
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int pix = dc.div(id), c = (id - pix * dc.d) * V;
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
        sycl::vec<sycl::half, V> v;
        v.load(0, src + ((n * OH + sy) * OW + sx) * CS + c);
#pragma unroll
        for (int j = 0; j < V; ++j)
          acc[j] += float(sycl::half(
              float(v[j]) *
              (div == 4.f       ? .25f
                   : div == 6.f ? 1.f / 6.f
                   : div == 9.f ? 1.f / 9.f
                                : 1.f / div)));
      }
    sycl::vec<sycl::half, V> out;
#pragma unroll
    for (int j = 0; j < V; ++j)
      out[j] = sycl::half(acc[j]);
    store<V>(dst + id * V, out);
  }
};
template <int K, int S, int P, bool Include, int V>
void launch(at::Tensor go, at::Tensor out) {
  if (out.size(1) % 32 != 0) {
    int C = out.size(1), H = out.size(2), W = out.size(3),
        count = out.numel() / V;
    c10::xpu::getCurrentXPUStream().queue().parallel_for(
        sycl::nd_range<1>((count + 255) / 256 * 256, 256),
        Linear<K, S, P, Include, V>{
            reinterpret_cast<const sycl::half*>(go.data_ptr()),
            reinterpret_cast<sycl::half*>(out.data_ptr()),
            C,
            H,
            W,
            int(go.size(2)),
            int(go.size(3)),
            int(go.stride(3)),
            count,
            Div(C / V),
            Div(W),
            Div(H)});
    return;
  }
  int C = out.size(1), H = out.size(2), W = out.size(3), tx = (W + 1) / 2,
      ty = (H + 1) / 2;
  int count = out.size(0) * tx * ty * (C / V);
  auto& q = c10::xpu::getCurrentXPUStream().queue();
  q.parallel_for(
      sycl::nd_range<1>((count + 255) / 256 * 256, 256),
      Tile<K, S, P, Include, V>{
          reinterpret_cast<const sycl::half*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          int(go.stride(3)),
          tx,
          ty,
          count,
          Div(C / V),
          Div(tx),
          Div(ty)});
}
template <int K, int S, int P, bool Include>
struct Planar {
  const sycl::half* src;
  sycl::half* dst;
  int H, W, OH, OW, total;
  Div dp, dw;
  void operator()(sycl::nd_item<1> it) const {
    int start = it.get_global_linear_id() * 8;
    if (start >= total)
      return;
    sycl::vec<sycl::half, 8> v;
    if constexpr (K == 2) {
      int nc = dp.div(start), pos = start - nc * H * W, y = dw.div(pos),
          x = pos - y * W;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        sycl::half a = sycl::half(
            float(src[nc * OH * OW + (y / 2) * OW + x / 2 + j]) * .25f);
        v[j * 2] = a;
        v[j * 2 + 1] = a;
      }
      store<8>(dst + start, v);
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
          acc += float(sycl::half(
              float(src[nc * OH * OW + sy * OW + sx]) *
              (div == 4.f       ? .25f
                   : div == 6.f ? 1.f / 6.f
                                : 1.f / 9.f)));
        }
      v[j] = sycl::half(acc);
    }
    store<8>(dst + start, v);
  }
};
template <int K, int S, int P, bool Include>
void planar_launch(at::Tensor go, at::Tensor out) {
  int H = out.size(2), W = out.size(3), n = out.numel();
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((n + 2047) / 2048 * 256, 256),
      Planar<K, S, P, Include>{
          reinterpret_cast<const sycl::half*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          n,
          Div(H * W),
          Div(W)});
}

template <int K, int S, int P, bool Include>
struct PackedLinear {
  const sycl::half* src;
  sycl::half* dst;
  int C, H, W, OH, OW, CS, total;
  Div dc, dw, dh;
  void operator()(sycl::nd_item<1> it) const {
    int start = it.get_global_linear_id() * 8;
    if (start >= total)
      return;
    int pixel = dc.div(start), c = start - pixel * C, row = dw.div(pixel),
        x = pixel - row * W, n = dh.div(row), y = row - n * H;
    sycl::vec<sycl::half, 8> result;
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
          sycl::vec<sycl::half, 2> v;
          v.load(0, src + ((nn * OH + sy) * OW + sx) * CS + ch);
          a += float(sycl::half(float(v[0]) * inv));
          b += float(sycl::half(float(v[1]) * inv));
        }
      result[pair * 2] = sycl::half(a);
      result[pair * 2 + 1] = sycl::half(b);
    }
    store<8>(dst + start, result);
  }
};
template <int K, int S, int P, bool Include>
void packed_launch(at::Tensor go, at::Tensor out) {
  int n = out.numel(), C = out.size(1), H = out.size(2), W = out.size(3);
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((n + 2047) / 2048 * 256, 256),
      PackedLinear<K, S, P, Include>{
          reinterpret_cast<const sycl::half*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          int(go.stride(3)),
          n,
          Div(C),
          Div(W),
          Div(H)});
}

template <int K, int S, int P, bool Include, int V>
struct WideTile {
  const sycl::half* src;
  sycl::half* dst;
  int C, H, W, OH, OW, CS, tx, ty, count;
  Div dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const {
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int q = dc.div(id), c = (id - q * dc.d) * V;
    int r = dx.div(q), x = (q - r * dx.d) * 4;
    int n = dy.div(r), y = (r - n * dy.d) * 2;
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
          sycl::vec<sycl::half, V> v;
          v.load(0, src + ((n * OH + sy) * OW + sx) * CS + c);
          float terms[V];
#pragma unroll
          for (int j = 0; j < V; ++j)
            terms[j] = float(sycl::half(float(v[j]) / divisor));
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
          sycl::vec<sycl::half, V> v;
#pragma unroll
          for (int j = 0; j < V; ++j)
            v[j] = sycl::half(acc[yy * 4 + xx][j]);
          store<V>(dst + ((n * H + y + yy) * W + x + xx) * C + c, v);
        }
      }
    }
  }
};

template <int K, int S, int P, bool Include>
void wide_launch(at::Tensor go, at::Tensor out) {
  int C = out.size(1), H = out.size(2), W = out.size(3), tx = (W + 3) / 4,
      ty = (H + 1) / 2;
  int count = out.size(0) * tx * ty * (C / 4);
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 255) / 256 * 256, 256),
      WideTile<K, S, P, Include, 4>{
          reinterpret_cast<const sycl::half*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          int(go.stride(3)),
          tx,
          ty,
          count,
          Div(C / 4),
          Div(tx),
          Div(ty)});
}
template <int V>
struct StrideTwo {
  const sycl::half* src;
  sycl::half* dst;
  int C, H, W, OH, OW, CS, count;
  Div dc, dw, dh;
  sycl::vec<float, V> get(int n, int y, int x, int c) const {
    sycl::vec<sycl::half, V> h;
    h.load(0, src + ((n * OH + y) * OW + x) * CS + c);
    sycl::vec<float, V> f;
#pragma unroll
    for (int j = 0; j < V; j++)
      f[j] = float(sycl::half(float(h[j]) * (1.f / 9.f)));
    return f;
  }
  void operator()(sycl::nd_item<1> it) const {
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int pix = dc.div(id), c = (id - pix * dc.d) * V, row = dw.div(pix),
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
    sycl::vec<sycl::half, V> out;
#pragma unroll
    for (int j = 0; j < V; j++)
      out[j] = sycl::half(sum[j]);
    store<V>(dst + id * V, out);
  }
};
template <int V>
void stride_two_launch(at::Tensor go, at::Tensor out) {
  int C = out.size(1), H = out.size(2), W = out.size(3),
      count = out.numel() / V;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 255) / 256 * 256, 256),
      StrideTwo<V>{
          reinterpret_cast<const sycl::half*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          H,
          W,
          int(go.size(2)),
          int(go.size(3)),
          int(go.stride(3)),
          count,
          Div(C / V),
          Div(W),
          Div(H)});
}

struct Esimd7Lean {
  const sycl::half* src;
  sycl::half* dst;
  int pairs;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace esimd = sycl::ext::intel::esimd;
    int id = it.get_global_linear_id();
    if (id >= pairs)
      return;
    esimd::simd<uint32_t, 416> idx(0, 1);
    auto pos = idx % 49;
    auto y = pos / 7;
    auto x = pos % 7;
    esimd::simd<sycl::half, 416> h;
    if (id + 1 < pairs) {
      h.select<256, 1>(0) = esimd::block_load<sycl::half, 256>(
          src + id * 392, esimd::properties{esimd::alignment<4>});
      h.select<128, 1>(256) = esimd::block_load<sycl::half, 128>(
          src + id * 392 + 256, esimd::properties{esimd::alignment<4>});
      h.select<32, 1>(384) = esimd::block_load<sycl::half, 32>(
          src + id * 392 + 384, esimd::properties{esimd::alignment<4>});
    } else
      h = esimd::gather<sycl::half, 416>(src + id * 392, idx * 2, idx < 392);
    esimd::simd<float, 416> v = h, inv = 1.f / 9.f;
    inv.merge(1.f / 6.f, (y == 0) | (y == 6) | (x == 0) | (x == 6));
    inv.merge(.25f, ((y == 0) | (y == 6)) & ((x == 0) | (x == 6)));
    esimd::simd<sycl::half, 416> norm = v * inv;
    v = norm;
    esimd::simd<uint16_t, 416> il = idx - 1, ir = idx + 1;
    il.merge(0, idx == 0);
    ir.merge(415, idx == 415);
    esimd::simd<float, 416> left = 0, right = 0;
    left.select<415, 1>(1) = v.select<415, 1>(0);
    right.select<415, 1>(0) = v.select<415, 1>(1);
    left.merge(0, x == 0);
    right.merge(0, x == 6);
    esimd::simd<float, 416> hor = (left + v) + right;
    esimd::simd<uint16_t, 416> iu = idx - 7, idn = idx + 7;
    iu.merge(0, idx < 7);
    idn.merge(415, idx >= 409);
    esimd::simd<float, 416> up = 0, down = 0;
    up.select<409, 1>(7) = hor.select<409, 1>(0);
    down.select<409, 1>(0) = hor.select<409, 1>(7);
    up.merge(0, y == 0);
    down.merge(0, y == 6);
    esimd::simd<sycl::half, 416> out = (up + hor) + down;
    esimd::simd<uint32_t, 208> packed = out.bit_cast_view<uint32_t>();
    esimd::simd<uint32_t, 208> oi(0, 1);
    auto target = reinterpret_cast<uint32_t*>(dst) + id * 196;
    esimd::block_store<uint32_t, 128>(
        target,
        packed.select<128, 1>(0),
        esimd::properties{esimd::alignment<4>});
    esimd::block_store<uint32_t, 64>(
        target + 128,
        packed.select<64, 1>(128),
        esimd::properties{esimd::alignment<4>});
    esimd::block_store<uint32_t, 4>(
        target + 192,
        packed.select<4, 1>(192),
        esimd::properties{esimd::alignment<4>});
  }
};
void lean7_launch(at::Tensor go, at::Tensor out) {
  int pairs = out.numel() / 392;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((pairs + 63) / 64 * 64, 64),
      Esimd7Lean{
          reinterpret_cast<const sycl::half*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          pairs});
}
template <int V>
struct DenseTile {
  const sycl::half* src;
  sycl::half* dst;
  int C, H, W, count;
  Div dc, dx, dy;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t z = (uint64_t(id) * dc.m) >> 32,
             q = dc.d == 1 ? id : z + uint32_t(id - z * dc.d >= dc.d);
    int c = (id - q * dc.d) * V;
    z = (uint64_t(q) * dx.m) >> 32;
    uint32_t r = dx.d == 1 ? q : z + uint32_t(q - z * dx.d >= dx.d);
    int x = (q - r * dx.d) * 2;
    z = (uint64_t(r) * dy.m) >> 32;
    int n = dy.d == 1 ? r : z + uint32_t(r - z * dy.d >= dy.d),
        y = (r - n * dy.d) * 2;
    e::simd<float, V> a = 0, b = 0, cc = 0, d = 0;
#pragma unroll
    for (int yy = 0; yy < 4; yy++) {
#pragma unroll
      for (int xx = 0; xx < 4; xx++) {
        int sy = y - 1 + yy, sx = x - 1 + xx;
        if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
          e::simd<sycl::half, V> h = e::block_load<sycl::half, V>(
              src + ((n * H + sy) * W + sx) * C + c,
              e::properties{e::alignment<16>});
          e::simd<float, V> v = h;
          h = v * (1.f / 9.f);
          v = h;
          if (yy < 3 && xx < 3)
            a += v;
          if (yy < 3 && xx > 0)
            b += v;
          if (yy > 0 && xx < 3)
            cc += v;
          if (yy > 0 && xx > 0)
            d += v;
        }
      }
    }
    e::simd<sycl::half, V> ha = a, hb = b, hc = cc, hd = d;
    auto target = dst + ((n * H + y) * W + x) * C + c;
    e::block_store<sycl::half, V>(target, ha, e::properties{e::alignment<16>});
    if (x + 1 < W)
      e::block_store<sycl::half, V>(
          target + C, hb, e::properties{e::alignment<16>});
    if (y + 1 < H) {
      e::block_store<sycl::half, V>(
          target + W * C, hc, e::properties{e::alignment<16>});
      if (x + 1 < W)
        e::block_store<sycl::half, V>(
            target + (W + 1) * C, hd, e::properties{e::alignment<16>});
    }
  }
};
template <int V>
void dense_launch(at::Tensor go, at::Tensor out, int wg) {
  int C = out.size(1), H = out.size(2), W = out.size(3), tx = (W + 1) / 2,
      ty = (H + 1) / 2, count = out.size(0) * tx * ty * (C / V);
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + wg - 1) / wg * wg, wg),
      DenseTile<V>{
          reinterpret_cast<const sycl::half*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          H,
          W,
          count,
          Div(C / V),
          Div(tx),
          Div(ty)});
}
template <int C, int V>
struct GappedS2 {
  const sycl::half* src;
  sycl::half* dst;
  int N, CS, count;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int n = id / 49, sy = (id % 49) / 7, sx = id % 7;
    e::simd<float, V> v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int yy = sy + j / 2, xx = sx + j % 2;
      e::simd<sycl::half, V> h = 0;
      if (yy < 7 && xx < 7) {
        int pix = (n * 7 + yy) * 7 + xx;
        if (pix + 1 < N * 49)
          h = e::block_load<sycl::half, V>(
              src + pix * CS, e::properties{e::alignment<16>});
        else {
          e::simd<uint32_t, V> ix(0, 1);
          h = e::gather<sycl::half, V>(src + pix * CS, ix * 2, ix < C);
        }
      }
      e::simd<float, V> f = h;
      h = f * (1.f / 9.f);
      v[j] = h;
    }
    e::simd<float, V> a = v[0], b = v[0] + v[1], cc = v[0] + v[2],
                      d = ((v[0] + v[1]) + v[2]) + v[3];
    e::simd<sycl::half, C * 2> top, bottom;
    top.template select<C, 1>(0) = a.template select<C, 1>(0);
    top.template select<C, 1>(C) = b.template select<C, 1>(0);
    bottom.template select<C, 1>(0) = cc.template select<C, 1>(0);
    bottom.template select<C, 1>(C) = d.template select<C, 1>(0);
    e::simd<uint32_t, C> ta = top.template bit_cast_view<uint32_t>(),
                         tb = bottom.template bit_cast_view<uint32_t>();
    auto target = reinterpret_cast<uint32_t*>(
        dst + ((n * 14 + sy * 2) * 14 + sx * 2) * C);
#pragma unroll
    for (int j = 0; j < C / 16; j++) {
      e::block_store<uint32_t, 16>(
          target + j * 16,
          ta.template select<16, 1>(j * 16),
          e::properties{e::alignment<16>});
      e::block_store<uint32_t, 16>(
          target + 7 * C + j * 16,
          tb.template select<16, 1>(j * 16),
          e::properties{e::alignment<16>});
    }
  }
};
template <int C, int V>
void gap_launch(at::Tensor go, at::Tensor out, int wg) {
  int N = out.size(0), count = N * 49;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + wg - 1) / wg * wg, wg),
      GappedS2<C, V>{
          reinterpret_cast<const sycl::half*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          N,
          int(go.stride(3)),
          count});
}
struct WideGap104 {
  const sycl::half* src;
  sycl::half* dst;
  int N, CS, count;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    int id = it.get_global_linear_id();
    if (id >= count)
      return;
    int n = id / 98, sy = (id % 98) / 7, sx = id % 7 * 2;
    e::simd<float, 128> v[6];
#pragma unroll
    for (int j = 0; j < 6; j++) {
      int yy = sy + j / 3, xx = sx + j % 3;
      e::simd<sycl::half, 128> h = 0;
      if (yy < 14 && xx < 14) {
        int pix = (n * 14 + yy) * 14 + xx;
        if (pix + 1 < N * 196)
          h = e::block_load<sycl::half, 128>(
              src + pix * CS, e::properties{e::alignment<16>});
        else {
          e::simd<uint32_t, 128> ix(0, 1);
          h = e::gather<sycl::half, 128>(src + pix * CS, ix * 2, ix < 104);
        }
      }
      e::simd<float, 128> f = h;
      h = f * (1.f / 9.f);
      v[j] = h;
    }
    e::simd<sycl::half, 416> top, bottom;
#pragma unroll
    for (int x = 0; x < 2; x++) {
      e::simd<float, 128> a = v[x], b = v[x] + v[x + 1], c = v[x] + v[x + 3],
                          d = ((v[x] + v[x + 1]) + v[x + 3]) + v[x + 4];
      top.template select<104, 1>(x * 208) = a.template select<104, 1>(0);
      top.template select<104, 1>(x * 208 + 104) = b.template select<104, 1>(0);
      bottom.template select<104, 1>(x * 208) = c.template select<104, 1>(0);
      bottom.template select<104, 1>(x * 208 + 104) =
          d.template select<104, 1>(0);
    }
    e::simd<uint32_t, 208> ta = top.template bit_cast_view<uint32_t>(),
                           tb = bottom.template bit_cast_view<uint32_t>();
    auto target = reinterpret_cast<uint32_t*>(
        dst + ((n * 28 + sy * 2) * 28 + sx * 2) * 104);
    e::block_store<uint32_t, 128>(
        target, ta.template select<128, 1>(0), e::properties{e::alignment<16>});
    e::block_store<uint32_t, 128>(
        target + 1456,
        tb.template select<128, 1>(0),
        e::properties{e::alignment<16>});
    e::block_store<uint32_t, 64>(
        target + 128,
        ta.template select<64, 1>(128),
        e::properties{e::alignment<16>});
    e::block_store<uint32_t, 64>(
        target + 1456 + 128,
        tb.template select<64, 1>(128),
        e::properties{e::alignment<16>});
    e::block_store<uint32_t, 16>(
        target + 192,
        ta.template select<16, 1>(192),
        e::properties{e::alignment<16>});
    e::block_store<uint32_t, 16>(
        target + 1456 + 192,
        tb.template select<16, 1>(192),
        e::properties{e::alignment<16>});
  }
};
void wide_gap104_launch(at::Tensor go, at::Tensor out, int wg) {
  int N = out.size(0), count = N * 98;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + wg - 1) / wg * wg, wg),
      WideGap104{
          reinterpret_cast<const sycl::half*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          N,
          int(go.stride(3)),
          count});
}
template <int W, int R, int CT>
struct RowStrip {
  const uint32_t* src;
  sycl::half* dst;
  int C, count;
  Div dt, dc;
  static constexpr int HW = W * W, OW = W / 2, OH = W / 2, NR = (W + R - 1) / R,
                       PL = ((R / 2 + 1) * OW + 31) / 32 * 32, CP = CT / 2,
                       OUT = (W * R + 31) / 32 * 32;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = (uint64_t(id) * dt.m) >> 32;
    int nc = dt.d == 1 ? id : q + uint32_t(id - q * dt.d >= dt.d),
        tile = id - nc * NR;
    q = (uint64_t(nc) * dc.m) >> 32;
    int n = dc.d == 1 ? nc : q + uint32_t(nc - q * dc.d >= dc.d),
        cb = (nc - n * dc.d) * CT, sy0 = tile * (R / 2), y0 = tile * R;
    e::simd<uint32_t, CP * PL> raw;
#pragma unroll
    for (int b = 0; b < PL / 32; b++) {
      auto t = e::load_2d<uint32_t, CP, 32, 1, true>(
          src + n * OH * OW * (C / 2),
          C * 2 - 1,
          OH * OW - 1,
          C * 2 - 1,
          cb / 2,
          sy0 * OW + b * 32);
#pragma unroll
      for (int cp = 0; cp < CP; cp++)
        raw.template select<32, 1>(cp * PL + b * 32) =
            t.template select<32, 1>(cp * 32);
    }
#pragma unroll
    for (int ch = 0; ch < CT; ch++) {
      e::simd<uint32_t, PL> words = raw.template select<PL, 1>(ch / 2 * PL);
      e::simd<sycl::half, PL> h =
          words.template bit_cast_view<sycl::half>().template select<PL, 2>(
              ch % 2);
      e::simd<float, PL> v = h;
      h = v * (1.f / 9.f);
      v = h;
      e::simd<sycl::half, OUT> out;
#pragma unroll
      for (int ry = 0; ry < R / 2; ry++) {
        e::simd<float, OW> a = v.template select<OW, 1>(ry * OW),
                           c = v.template select<OW, 1>((ry + 1) * OW), b = 0,
                           d = 0;
        b.template select<OW - 1, 1>(0) = a.template select<OW - 1, 1>(1);
        d.template select<OW - 1, 1>(0) = c.template select<OW - 1, 1>(1);
        if (sy0 + ry + 1 >= OH)
          c = 0, d = 0;
        e::simd<sycl::half, OW> ha = a, hab = a + b, hac = a + c,
                                habcd = ((a + b) + c) + d;
        out.template select<OW, 2>(ry * 2 * W) = ha;
        out.template select<OW, 2>(ry * 2 * W + 1) = hab;
        out.template select<OW, 2>((ry * 2 + 1) * W) = hac;
        out.template select<OW, 2>((ry * 2 + 1) * W + 1) = habcd;
      }
      auto target =
          reinterpret_cast<uint32_t*>(dst + (n * C + cb + ch) * HW + y0 * W);
      e::simd<uint32_t, OUT / 2> packed =
          out.template bit_cast_view<uint32_t>();
#pragma unroll
      for (int b = 0; b < OUT / 32; b++) {
        e::simd<uint32_t, 16> val = packed.template select<16, 1>(b * 16);
        if (y0 * W + b * 32 + 32 <= HW && b * 32 + 32 <= W * R)
          e::block_store<uint32_t, 16>(
              target + b * 16, val, e::properties{e::alignment<4>});
        else {
          e::simd<uint32_t, 16> idx(0, 1);
          e::scatter<uint32_t, 16>(
              target + b * 16,
              idx * 4,
              val,
              (y0 * W + b * 32 + idx * 2 < HW) & (b * 32 + idx * 2 < W * R));
        }
      }
    }
  }
};
template <int W, int R, int CT>
void rs_launch(at::Tensor go, at::Tensor out) {
  int C = out.size(1), ct = C / CT, tiles = (W + R - 1) / R,
      count = out.size(0) * ct * tiles;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + 15) / 16 * 16, 16),
      RowStrip<W, R, CT>{
          reinterpret_cast<const uint32_t*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          count,
          Div(tiles),
          Div(ct)});
}
template <int R, int CT>
void rs_dispatch(at::Tensor go, at::Tensor out) {
  switch (out.size(3)) {
    case 14:
      rs_launch<14, R, CT>(go, out);
      break;
    case 16:
      rs_launch<16, R, CT>(go, out);
      break;
    case 28:
      rs_launch<28, R, CT>(go, out);
      break;
    case 32:
      rs_launch<32, R, CT>(go, out);
      break;
    case 56:
      rs_launch<56, R, CT>(go, out);
      break;
    case 64:
      rs_launch<64, R, CT>(go, out);
      break;
  }
}

template <int W, int CT>
struct WholeStrip {
  const uint32_t* src;
  sycl::half* dst;
  int C, count;
  Div dt, dc;
  static constexpr int R = W, HW = W * W, OW = W / 2, OH = W / 2,
                       NR = (W + R - 1) / R,
                       PL = ((R / 2 + 1) * OW + 31) / 32 * 32, CP = CT / 2,
                       OUT = (W * W * CT + 127) / 128 * 128;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = (uint64_t(id) * dt.m) >> 32;
    int nc = dt.d == 1 ? id : q + uint32_t(id - q * dt.d >= dt.d),
        tile = id - nc * NR;
    q = (uint64_t(nc) * dc.m) >> 32;
    int n = dc.d == 1 ? nc : q + uint32_t(nc - q * dc.d >= dc.d),
        cb = (nc - n * dc.d) * CT, sy0 = tile * (R / 2), y0 = tile * R;
    e::simd<uint32_t, CP * PL> raw;
#pragma unroll
    for (int b = 0; b < PL / 32; b++) {
      auto t = e::load_2d<uint32_t, CP, 32, 1, true>(
          src + n * OH * OW * (C / 2),
          C * 2 - 1,
          OH * OW - 1,
          C * 2 - 1,
          cb / 2,
          sy0 * OW + b * 32);
#pragma unroll
      for (int cp = 0; cp < CP; cp++)
        raw.template select<32, 1>(cp * PL + b * 32) =
            t.template select<32, 1>(cp * 32);
    }
    e::simd<sycl::half, OUT> out = 0;
#pragma unroll
    for (int ch = 0; ch < CT; ch++) {
      e::simd<uint32_t, PL> words = raw.template select<PL, 1>(ch / 2 * PL);
      e::simd<sycl::half, PL> h =
          words.template bit_cast_view<sycl::half>().template select<PL, 2>(
              ch % 2);
      e::simd<float, PL> v = h;
      h = v * (1.f / 9.f);
      v = h;

#pragma unroll
      for (int ry = 0; ry < R / 2; ry++) {
        e::simd<float, OW> a = v.template select<OW, 1>(ry * OW),
                           c = v.template select<OW, 1>((ry + 1) * OW), b = 0,
                           d = 0;
        b.template select<OW - 1, 1>(0) = a.template select<OW - 1, 1>(1);
        d.template select<OW - 1, 1>(0) = c.template select<OW - 1, 1>(1);
        if (sy0 + ry + 1 >= OH)
          c = 0, d = 0;
        e::simd<sycl::half, OW> ha = a, hab = a + b, hac = a + c,
                                habcd = ((a + b) + c) + d;
        out.template select<OW, 2>(ch * HW + ry * 2 * W) = ha;
        out.template select<OW, 2>(ch * HW + ry * 2 * W + 1) = hab;
        out.template select<OW, 2>(ch * HW + (ry * 2 + 1) * W) = hac;
        out.template select<OW, 2>(ch * HW + (ry * 2 + 1) * W + 1) = habcd;
      }
    }
    auto target = reinterpret_cast<uint32_t*>(dst + (n * C + cb) * HW);
    e::simd<uint32_t, OUT / 2> packed = out.template bit_cast_view<uint32_t>();
#pragma unroll
    for (int b = 0; b < OUT / 128; b++) {
      e::simd<uint32_t, 64> val = packed.template select<64, 1>(b * 64);
      if (b * 128 + 128 <= HW * CT)
        e::block_store<uint32_t, 64>(
            target + b * 64, val, e::properties{e::alignment<4>});
      else {
        e::simd<uint32_t, 64> idx(0, 1);
        e::scatter<uint32_t, 64>(
            target + b * 64, idx * 4, val, b * 128 + idx * 2 < HW * CT);
      }
    }
  }
};
template <int W, int CT>
void ws_launch(at::Tensor go, at::Tensor out, int wg) {
  int C = out.size(1), ct = C / CT, count = out.size(0) * ct;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + wg - 1) / wg * wg, wg),
      WholeStrip<W, CT>{
          reinterpret_cast<const uint32_t*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          count,
          Div(1),
          Div(ct)});
}
template <int W, int R, int CT, int CB, int TB>
struct GroupedS1 {
  const uint32_t* src;
  sycl::half* dst;
  int C, count;
  Div dt, dc;
  static constexpr int HW = W * W, NR = (W + R - 1) / R,
                       PL = ((R + 2) * W + 31) / 32 * 32, CP = CT / 2,
                       STRIP = W * R, COMB = R == W,
                       OUT = ((COMB ? CT : 1) * STRIP + 127) / 128 * 128;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t group = id / CB, inner = id % CB,
             q = (uint64_t(group) * dt.m) >> 32;
    uint32_t channel_group =
                 dt.d == 1 ? group : q + uint32_t(group - q * dt.d >= dt.d),
             tile = group - channel_group * NR, nc = channel_group * CB;
    q = (uint64_t(nc) * dc.m) >> 32;
    int n = dc.d == 1 ? nc : q + uint32_t(nc - q * dc.d >= dc.d),
        cb = (nc - n * dc.d + inner) * CT, y0 = tile * R;
    e::simd<uint32_t, CP * PL> raw;
#pragma unroll
    for (int b = 0; b < PL / 32; b++) {
      auto t = e::load_2d<uint32_t, CP, 32, 1, true>(
          src + n * HW * (C / 2),
          C * 2 - 1,
          HW - 1,
          C * 2 - 1,
          cb / 2,
          (y0 - 1) * W + b * 32);
#pragma unroll
      for (int cp = 0; cp < CP; cp++)
        raw.template select<32, 1>(cp * PL + b * 32) =
            t.template select<32, 1>(cp * 32);
    }
    e::simd<sycl::half, OUT> out = 0;
    e::simd<int, PL> index(0, 1);
    auto yy = index / W + y0 - 1, xx = index % W;
    e::simd<float, PL> inv = 1.f / 9.f;
    auto ey = (yy == 0) | (yy == W - 1), ex = (xx == 0) | (xx == W - 1);
    inv.merge(1.f / 6.f, ey | ex);
    inv.merge(.25f, ey & ex);
#pragma unroll
    for (int ch = 0; ch < CT; ch++) {
      e::simd<uint32_t, PL> words = raw.template select<PL, 1>(ch / 2 * PL);
      e::simd<sycl::half, PL> h =
          words.template bit_cast_view<sycl::half>().template select<PL, 2>(
              ch % 2);
      e::simd<float, PL> v = h;
      h = v * inv;
      v = h;
#pragma unroll
      for (int ry = 0; ry < R; ry++) {
        e::simd<float, W> acc = 0;
#pragma unroll
        for (int dy = 0; dy < 3; dy++) {
          e::simd<float, W> mid = v.template select<W, 1>((ry + dy) * W),
                            left = 0, right = 0;
          left.template select<W - 1, 1>(1) = mid.template select<W - 1, 1>(0);
          right.template select<W - 1, 1>(0) = mid.template select<W - 1, 1>(1);
          acc += left;
          acc += mid;
          acc += right;
        }
        e::simd<sycl::half, W> res = acc;
        out.template select<W, 1>((COMB ? ch * STRIP : 0) + ry * W) = res;
      }
      if constexpr (!COMB) {
        auto target =
            reinterpret_cast<uint32_t*>(dst + (n * C + cb + ch) * HW + y0 * W);
        e::simd<uint32_t, OUT / 2> packed =
            out.template bit_cast_view<uint32_t>();
#pragma unroll
        for (int b = 0; b < OUT / 128; b++) {
          e::simd<uint32_t, 64> val = packed.template select<64, 1>(b * 64);
          if (b * 128 + 128 <= STRIP && y0 * W + b * 128 + 128 <= HW)
            e::block_store<uint32_t, 64>(
                target + b * 64, val, e::properties{e::alignment<4>});
          else {
            e::simd<uint32_t, 64> idx(0, 1);
            e::scatter<uint32_t, 64>(
                target + b * 64,
                idx * 4,
                val,
                (b * 128 + idx * 2 < STRIP) &
                    (y0 * W + b * 128 + idx * 2 < HW));
          }
        }
      }
    }
    if constexpr (COMB) {
      auto target = reinterpret_cast<uint32_t*>(dst + (n * C + cb) * HW);
      e::simd<uint32_t, OUT / 2> packed =
          out.template bit_cast_view<uint32_t>();
#pragma unroll
      for (int b = 0; b < OUT / 128; b++) {
        e::simd<uint32_t, 64> val = packed.template select<64, 1>(b * 64);
        if (b * 128 + 128 <= HW * CT)
          e::block_store<uint32_t, 64>(
              target + b * 64, val, e::properties{e::alignment<4>});
        else {
          e::simd<uint32_t, 64> idx(0, 1);
          e::scatter<uint32_t, 64>(
              target + b * 64, idx * 4, val, b * 128 + idx * 2 < HW * CT);
        }
      }
    }
  }
};
template <int W, int R, int CT, int CB, int TB>
void grouped_launch(at::Tensor go, at::Tensor out, int wg = 16) {
  int C = out.size(1), ct = C / CT, tiles = (W + R - 1) / R,
      count = out.size(0) * ct * tiles;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + wg - 1) / wg * wg, wg),
      GroupedS1<W, R, CT, CB, TB>{
          reinterpret_cast<const uint32_t*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          count,
          Div(tiles),
          Div(ct)});
}
template <int Count, int Lead, int N>
ESIMD_INLINE void constant_sector_store(
    uint32_t* target,
    sycl::ext::intel::esimd::simd<uint32_t, N> packed) SYCL_ESIMD_FUNCTION {
  namespace e = sycl::ext::intel::esimd;
  constexpr int Full = (Count - Lead) / 16, Tail = Count - Lead - Full * 16;
  if constexpr (Lead > 0) {
    e::simd<uint32_t, 16> idx(0, 1);
    e::scatter<uint32_t, 16>(
        target, idx * 4, packed.template select<16, 1>(0), idx < Lead);
  }
#pragma unroll
  for (int b = 0; b < Full / 8; b++)
    e::block_store<uint32_t, 128>(
        target + Lead + b * 128,
        packed.template select<128, 1>(Lead + b * 128),
        e::properties{e::alignment<16>});
#pragma unroll
  for (int b = Full / 8 * 8; b < Full; b++)
    e::block_store<uint32_t, 16>(
        target + Lead + b * 16,
        packed.template select<16, 1>(Lead + b * 16),
        e::properties{e::alignment<16>});
  if constexpr (Tail > 0) {
    constexpr int Off = Lead + Full * 16;
    e::simd<uint32_t, 16> idx(0, 1);
    e::scatter<uint32_t, 16>(
        target + Off, idx * 4, packed.template select<16, 1>(Off), idx < Tail);
  }
}
template <int W, int R, int CT>
struct CompactS1 {
  const uint32_t* src;
  sycl::half* dst;
  int C, count;
  Div dt, dc;
  static constexpr int HW = W * W, NR = (W + R - 1) / R,
                       PL = ((R + 2) * W + 31) / 32 * 32, CP = CT / 2,
                       STRIP = W * R, COMB = R == W,
                       OUT = ((COMB ? CT : 1) * STRIP + 127) / 128 * 128;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = (uint64_t(id) * dt.m) >> 32;
    int nc = dt.d == 1 ? id : q + uint32_t(id - q * dt.d >= dt.d),
        tile = id - nc * NR;
    q = (uint64_t(nc) * dc.m) >> 32;
    int n = dc.d == 1 ? nc : q + uint32_t(nc - q * dc.d >= dc.d),
        cb = (nc - n * dc.d) * CT, y0 = tile * R;
    e::simd<uint32_t, CP * PL> raw;
#pragma unroll
    for (int b = 0; b < PL / 32; b++) {
      auto t = e::load_2d<uint32_t, CP, 32, 1, true>(
          src + n * HW * (C / 2),
          C * 2 - 1,
          HW - 1,
          C * 2 - 1,
          cb / 2,
          y0 * W + b * 32);
#pragma unroll
      for (int cp = 0; cp < CP; cp++)
        raw.template select<32, 1>(cp * PL + b * 32) =
            t.template select<32, 1>(cp * 32);
    }
    e::simd<sycl::half, OUT> out;
    e::simd<int, HW> index(0, 1);
    auto yy = index / W, xx = index % W;
    e::simd<float, HW> inv = 1.f / 9.f;
    auto ey = (yy == 0) | (yy == W - 1), ex = (xx == 0) | (xx == W - 1);
    inv.merge(1.f / 6.f, ey | ex);
    inv.merge(.25f, ey & ex);
#pragma unroll
    for (int ch = 0; ch < CT; ch++) {
      e::simd<uint32_t, HW> words = raw.template select<HW, 1>(ch / 2 * PL);
      e::simd<sycl::half, HW> h =
          words.template bit_cast_view<sycl::half>().template select<HW, 2>(
              ch % 2);
      e::simd<float, HW> v = h;
      h = v * inv;
      v = h;
      e::simd<float, HW> left = 0, right = 0;
      left.template select<HW - 1, 1>(1) = v.template select<HW - 1, 1>(0);
      right.template select<HW - 1, 1>(0) = v.template select<HW - 1, 1>(1);
      left.merge(0, xx == 0);
      right.merge(0, xx == W - 1);
      e::simd<float, HW> hor = (left + v) + right, up = 0, down = 0;
      up.template select<HW - W, 1>(W) = hor.template select<HW - W, 1>(0);
      down.template select<HW - W, 1>(0) = hor.template select<HW - W, 1>(W);
      e::simd<sycl::half, HW> res = (up + hor) + down;
      out.template select<HW, 1>(ch * HW) = res;
      if constexpr (!COMB) {
        auto target =
            reinterpret_cast<uint32_t*>(dst + (n * C + cb + ch) * HW + y0 * W);
        e::simd<uint32_t, OUT / 2> packed =
            out.template bit_cast_view<uint32_t>();
#pragma unroll
        for (int b = 0; b < OUT / 128; b++) {
          e::simd<uint32_t, 64> val = packed.template select<64, 1>(b * 64);
          if (b * 128 + 128 <= STRIP && y0 * W + b * 128 + 128 <= HW)
            e::block_store<uint32_t, 64>(
                target + b * 64, val, e::properties{e::alignment<4>});
          else {
            e::simd<uint32_t, 64> idx(0, 1);
            e::scatter<uint32_t, 64>(
                target + b * 64,
                idx * 4,
                val,
                (b * 128 + idx * 2 < STRIP) &
                    (y0 * W + b * 128 + idx * 2 < HW));
          }
        }
      }
    }
    if constexpr (COMB) {
      auto target = reinterpret_cast<uint32_t*>(dst) + (n * C + cb) * (HW / 2);
      e::simd<uint32_t, OUT / 2> packed =
          out.template bit_cast_view<uint32_t>();
      if constexpr (CT == 4) {
        if (id % 2)
          constant_sector_store<392, 8>(target, packed);
        else
          constant_sector_store<392, 0>(target, packed);
      } else {
        switch (id % 4) {
          case 0:
            constant_sector_store<196, 0>(target, packed);
            break;
          case 1:
            constant_sector_store<196, 12>(target, packed);
            break;
          case 2:
            constant_sector_store<196, 8>(target, packed);
            break;
          case 3:
            constant_sector_store<196, 4>(target, packed);
            break;
        }
      }
    }
  }
};
template <int W, int R, int CT>
void compact_launch(at::Tensor go, at::Tensor out, int wg = 16) {
  int C = out.size(1), ct = C / CT, tiles = (W + R - 1) / R,
      count = out.size(0) * ct * tiles;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + wg - 1) / wg * wg, wg),
      CompactS1<W, R, CT>{
          reinterpret_cast<const uint32_t*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          count,
          Div(tiles),
          Div(ct)});
}
template <int W, int R, int CT>
struct StreamPlanes {
  const uint32_t* src;
  sycl::half* dst;
  int C, count;
  Div dc;
  static constexpr int HW = W * W, PL = ((R + 2) * W + 31) / 32 * 32,
                       OUT = HW * CT;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    namespace e = sycl::ext::intel::esimd;
    uint32_t id = it.get_global_linear_id();
    if (id >= count)
      return;
    uint32_t q = (uint64_t(id) * dc.m) >> 32;
    int n = dc.d == 1 ? id : q + uint32_t(id - q * dc.d >= dc.d),
        cb = (id - n * dc.d) * CT;
    e::simd<sycl::half, OUT> out;
#pragma unroll
    for (int cp = 0; cp < CT / 2; cp++) {
#pragma unroll
      for (int tile = 0; tile < W / R; tile++) {
        int y0 = tile * R;
        e::simd<uint32_t, PL> raw;
#pragma unroll
        for (int b = 0; b < PL / 32; b++)
          raw.template select<32, 1>(b * 32) =
              e::load_2d<uint32_t, 1, 32, 1, true>(
                  src + n * HW * (C / 2),
                  C * 2 - 1,
                  HW - 1,
                  C * 2 - 1,
                  cb / 2 + cp,
                  (y0 - 1) * W + b * 32);
        e::simd<int, PL> index(0, 1);
        auto yy = index / W + y0 - 1, xx = index % W;
        e::simd<float, PL> inv = 1.f / 9.f;
        auto ey = (yy == 0) | (yy == W - 1), ex = (xx == 0) | (xx == W - 1);
        inv.merge(1.f / 6.f, ey | ex);
        inv.merge(.25f, ey & ex);
#pragma unroll
        for (int ch = 0; ch < 2; ch++) {
          e::simd<sycl::half, PL> h =
              raw.template bit_cast_view<sycl::half>().template select<PL, 2>(
                  ch);
          e::simd<float, PL> v = h;
          h = v * inv;
          v = h;
          e::simd<float, PL> left = 0, right = 0;
          left.template select<PL - 1, 1>(1) = v.template select<PL - 1, 1>(0);
          right.template select<PL - 1, 1>(0) = v.template select<PL - 1, 1>(1);
          left.merge(0, xx == 0);
          right.merge(0, xx == W - 1);
          e::simd<float, PL> hor = (left + v) + right;
#pragma unroll
          for (int ry = 0; ry < R; ry++) {
            e::simd<float, W> a = hor.template select<W, 1>(ry * W),
                              b = hor.template select<W, 1>((ry + 1) * W),
                              c = hor.template select<W, 1>((ry + 2) * W);
            e::simd<sycl::half, W> val = (a + b) + c;
            out.template select<W, 1>((cp * 2 + ch) * HW + (y0 + ry) * W) = val;
          }
        }
      }
    }
    e::simd<uint32_t, OUT / 2> packed = out.template bit_cast_view<uint32_t>();
    auto target = reinterpret_cast<uint32_t*>(dst) + (n * C + cb) * (HW / 2);
#pragma unroll
    for (int b = 0; b < OUT / 256; b++)
      e::block_store<uint32_t, 128>(
          target + b * 128,
          packed.template select<128, 1>(b * 128),
          e::properties{e::alignment<16>});
#pragma unroll
    for (int b = OUT / 256 * 8; b < OUT / 32; b++)
      e::block_store<uint32_t, 16>(
          target + b * 16,
          packed.template select<16, 1>(b * 16),
          e::properties{e::alignment<16>});
  }
};
template <int W, int R, int CT>
void stream_launch(at::Tensor go, at::Tensor out, int wg) {
  int C = out.size(1), ct = C / CT, count = out.size(0) * ct;
  c10::xpu::getCurrentXPUStream().queue().parallel_for(
      sycl::nd_range<1>((count + wg - 1) / wg * wg, wg),
      StreamPlanes<W, R, CT>{
          reinterpret_cast<const uint32_t*>(go.data_ptr()),
          reinterpret_cast<sycl::half*>(out.data_ptr()),
          C,
          count,
          Div(ct)});
}
struct ZeroInsert {
  const uint32_t* src;
  uint32_t* dst;
  int CP, H, W, OH, OW, total;
  Div dc, dw, dh;
  void operator()(sycl::nd_item<1> it) const {
    int step = it.get_global_range(0) * 4;
    for (int start = it.get_global_linear_id() * 4; start < total;
         start += step) {
      int pix = dc.div(start), ch = start - pix * CP, row = dw.div(pix),
          x = pix - row * W, n = dh.div(row), y = row - n * H;
      sycl::vec<uint32_t, 4> v(0);
      if (ch + 4 <= CP) {
        if (!(x % 2) && !(y % 2))
          v.load(0, src + ((n * OH + y / 2) * OW + x / 2) * CP + ch);
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
        v.store(0, dst + start);
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
            Div(C / 2),
            Div(W),
            Div(H)}});
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
          Div(C / 2),
          Div(W),
          Div(H)});
}

template <int K, int S, int P, bool Include>
void merged_launch(at::Tensor go, at::Tensor out) {
  const int C = out.size(1), H = out.size(2), W = out.size(3);
  if (out.stride(1) != 1) {
    if (go.stride(1) != 1) {
      if constexpr (K == 3 && S == 1 && !Include) {
        lean7_launch(go, out);
        return;
      }
      planar_launch<K, S, P, Include>(go, out);
      return;
    }
    if constexpr (K == 3 && S == 1) {
      if (W == 14)
        compact_launch<14, 14, 4>(go, out, 8);
      else if (W == 28)
        stream_launch<28, 4, 2>(go, out, 16);
      else
        grouped_launch<56, 4, 4, 8, 1>(go, out, 8);
      return;
    }
    if constexpr (K == 3 && S == 2) {
      if (W == 14)
        ws_launch<14, 8>(go, out, 16);
      else if (W == 28)
        ws_launch<28, 2>(go, out, 16);
      else
        rs_dispatch<8, 4>(go, out);
      return;
    }
    TORCH_INTERNAL_ASSERT(false, "unsupported mixed pool layout");
  }
  if constexpr (K == 3 && S == 2 && Include) {
    if (H == 28 && W == 28 && C == 104) {
      wide_gap104_launch(go, out, 32);
      return;
    }
    if (H == 14 && W == 14 && C == 112) {
      gap_launch<112, 128>(go, out, 32);
      return;
    }
    if (H == 14 && W == 14 && C == 208) {
      gap_launch<208, 256>(go, out, 32);
      return;
    }
    if (C % 32 != 0) {
      if (C % 8 == 0)
        stride_two_launch<8>(go, out);
      else
        stride_two_launch<4>(go, out);
      return;
    }
  }
  if constexpr (K == 3 && S == 1 && Include) {
    if (go.stride(3) == C && C % 256 == 0 && H == 35 && W == 35) {
      dense_launch<256>(go, out, 16);
      return;
    }
    if (go.stride(3) == C && C % 32 == 0 && W <= 17) {
      wide_launch<K, S, P, Include>(go, out);
      return;
    }
  }
  if constexpr (K == 1) {
    if (C % 32 != 0) {
      zero_launch(go, out, 2560);
      return;
    }
  }
  if (C % 8 != 0) {
    packed_launch<K, S, P, Include>(go, out);
    return;
  }
  launch<K, S, P, Include, 8>(go, out);
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
      input.scalar_type() != kHalf || go.scalar_type() != kHalf ||
      out.scalar_type() != kHalf || divisor || input.numel() == 0 || kh != kw ||
      sh != sw || ph != pw || !canUse32BitIndexMath(input) ||
      !canUse32BitIndexMath(go) || !canUse32BitIndexMath(out) ||
      out.sizes() != input.sizes()) {
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
  const auto arch =
      c10::xpu::getCurrentXPUStream()
          .queue()
          .get_device()
          .get_info<
              sycl::ext::oneapi::experimental::info::device::architecture>();
  using Architecture = sycl::ext::oneapi::experimental::architecture;
  if (arch != Architecture::intel_gpu_bmg_g21 &&
      arch != Architecture::intel_gpu_bmg_g31) {
    return false;
  }
  if (pool3 && sh == 1) {
    if (include)
      merged_launch<3, 1, 1, true>(go, out);
    else
      merged_launch<3, 1, 1, false>(go, out);
  } else if (pool3) {
    merged_launch<3, 2, 1, true>(go, out);
  } else if (pool2) {
    if (include)
      merged_launch<2, 2, 0, true>(go, out);
    else
      merged_launch<2, 2, 0, false>(go, out);
  } else {
    merged_launch<1, 2, 0, false>(go, out);
  }
  return true;
}
} // namespace at::native::xpu::bmg_avg_pool
