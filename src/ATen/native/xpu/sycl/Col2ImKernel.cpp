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

#include <ATen/Dispatch.h>
#include <comm/xpu_aten.h>

#include <ATen/native/im2col_shape_check.h>
#include <ATen/xpu/XPUContext.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/Col2ImKernel.h>

#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <vector>

namespace at::native::xpu {

// ---------------------------------------------------------------------------
// Baseline: one work-item per output pixel, per-batch launch (unchanged)
// ---------------------------------------------------------------------------

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void col2im_ff_kernel(
    const T* in_data,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* out_data) {
  auto itemId = syclext::this_work_item::get_nd_item<1>();
  auto in_ptr = in_data;
  auto out_ptr = out_data;
  auto id = itemId.get_global_id(0);
  if (id >= channels * height * width) {
    return;
  }

  T val = static_cast<T>(0);
  const int64_t w_im = id % width + pad_w;
  const int64_t h_im = (id / width) % height + pad_h;
  const int64_t c_im = id / (width * height);
  int64_t kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
  int64_t kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
  const int64_t w_col_start =
      (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
  const int64_t w_col_end = std::min(w_im / stride_w + 1, output_width);
  const int64_t h_col_start =
      (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
  const int64_t h_col_end = std::min(h_im / stride_h + 1, output_height);

  for (int64_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
    for (int64_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
      int64_t h_k = (h_im - h_col * stride_h);
      int64_t w_k = (w_im - w_col * stride_w);
      if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
        h_k /= dilation_h;
        w_k /= dilation_w;
        int64_t data_col_index =
            (((c_im * kernel_h + h_k) * kernel_w + w_k) * output_height + h_col) *
                output_width + w_col;
        val += in_ptr[data_col_index];
      }
    }
  }
  out_ptr[id] = static_cast<T>(val);
}

template <typename T>
static void col2im_launch_baseline(
    const T* data_col,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_im) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  auto total_threads = channels * width * height;
  int64_t local_range = at::xpu::getKernelMaxWorkGroupSize<col2im_ff_kernel<T>>();
  int64_t global_range = ((total_threads + local_range - 1) / local_range) * local_range;
  sycl_kernel_submit<col2im_ff_kernel<T>>(
      ::sycl::range<1>(global_range), ::sycl::range<1>(local_range), sycl_queue, 0,
      in_data, channels, height, width, output_height, output_width,
      kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, data_im);
}

// ---------------------------------------------------------------------------
// Candidate: one launch for the whole batch, SLM-staged tiles, 16 B contiguous output stores
// ---------------------------------------------------------------------------

namespace {

struct Magic {
  uint32_t m;
  uint32_t s;
};

inline Magic make_magic(uint32_t d) {
  uint32_t s = 0;
  while (s < 31 && (1u << s) < d) ++s;
  const uint64_t m = ((uint64_t(1) << 32) * ((uint64_t(1) << s) - d)) / d + 1;
  return Magic{static_cast<uint32_t>(m), s};
}

inline uint32_t mdiv(uint32_t n, Magic g) {
  return (sycl::mul_hi(n, g.m) + n) >> g.s;
}

struct Params {
  const char* in;
  char* out;
  int N, C, OH, OW, BH, BW, KH, KW, SH, SW, PH, PW, L, K2, CK;
  int layoutB, CB, HB, nCB, nHB, WG, TW, RSw, RWd;
  int PWfull, PWtail, PRVfull, PRVtail;
  int sKh, sKw, sOh, sOw, cstr;
  int PV;
  Magic mOW, mSH, mSW, mPRVfull, mPRVtail, mPWfull, mPWtail;
};

template <typename T, int CHT, int CWT>
struct TileKernel {
  static constexpr int SZ = sizeof(T);
  static constexpr int V = 16 / SZ;
  static constexpr int EPW = 16 / SZ;
  static constexpr int UNR = 4;
  static constexpr uint32_t INV = 1u << 28;
  using U = std::conditional_t<SZ == 2, uint16_t, uint32_t>;

  Params p;
  sycl::local_accessor<sycl::uint4, 1> tile;
  sycl::local_accessor<int, 1> tab;

  static T from_bits(U u) {
    if constexpr (SZ == 2) { T t; t.x = u; return t; }
    else { return sycl::bit_cast<float>(u); }
  }
  static U to_bits(T t) {
    if constexpr (SZ == 2) { return t.x; }
    else { return sycl::bit_cast<uint32_t>(t); }
  }

  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> it) const {
    const uint32_t lid = it.get_local_id(0);
    const uint32_t g = it.get_group(0);
    const uint32_t WG = p.WG;
    const uint32_t cb_i = g % p.nCB;
    const uint32_t tq = g / p.nCB;
    const uint32_t hb_i = tq % p.nHB;
    const uint32_t n = tq / p.nHB;
    const uint32_t c0 = cb_i * p.CB;
    const uint32_t CBc = sycl::min<uint32_t>(p.CB, p.C - c0);
    const uint32_t h0 = hb_i * p.HB;
    const uint32_t HBc = sycl::min<uint32_t>(p.HB, p.OH - h0);
    const bool lastH = hb_i + 1 == uint32_t(p.nHB);
    const bool lastC = cb_i + 1 == uint32_t(p.nCB);
    const uint32_t PRV = lastH ? p.PRVtail : p.PRVfull;
    const Magic mPRV = lastH ? p.mPRVtail : p.mPRVfull;
    const uint32_t ZOFF = uint32_t(p.TW) * EPW;
    const uint32_t OW = p.OW;

    uint32_t base0 = 0;
    uint32_t ohbase = 0;
    {
      const sycl::uint4* src;
      uint32_t NW;
      uint32_t row_lo = 0;
      if (!p.layoutB) {
        const size_t gs = (size_t(n) * p.CK + size_t(c0) * p.K2) * size_t(p.L) * SZ;
        const uint32_t mis = (reinterpret_cast<uintptr_t>(p.in) + gs) & 15u;
        src = reinterpret_cast<const sycl::uint4*>(p.in + (gs - mis));
        base0 = mis / SZ;
        NW = (mis + CBc * p.K2 * p.L * SZ + 15) >> 4;
      } else {
        const int t_lo = int(h0) + p.PH - (p.KH - 1);
        const uint32_t oh_lo = t_lo > 0 ? mdiv(uint32_t(t_lo + p.SH - 1), p.mSH) : 0u;
        const uint32_t oh_hi = sycl::min<uint32_t>(p.BH - 1, mdiv(h0 + HBc - 1 + p.PH, p.mSH));
        const uint32_t nrows = oh_hi >= oh_lo ? (oh_hi - oh_lo + 1) * p.BW : 0u;
        ohbase = oh_lo;
        row_lo = oh_lo * p.BW;
        const uint32_t PWc = lastC ? p.PWtail : p.PWfull;
        NW = nrows * PWc;
        src = reinterpret_cast<const sycl::uint4*>(p.in);
      }
      if (!p.layoutB) {
        for (uint32_t b = lid; b < NW; b += WG * UNR) {
          sycl::uint4 r[UNR];
#pragma unroll
          for (int u = 0; u < UNR; ++u) {
            const uint32_t i = b + u * WG;
            if (i < NW) r[u] = src[i];
          }
#pragma unroll
          for (int u = 0; u < UNR; ++u) {
            const uint32_t i = b + u * WG;
            if (i < NW) tile[i] = r[u];
          }
        }
      } else {
        const Magic mPW = lastC ? p.mPWtail : p.mPWfull;
        const uint32_t PWc = lastC ? p.PWtail : p.PWfull;
        const size_t cw0 = (size_t(c0) * p.K2 * SZ) >> 4;
        for (uint32_t b = lid; b < NW; b += WG * UNR) {
          sycl::uint4 r[UNR];
#pragma unroll
          for (int u = 0; u < UNR; ++u) {
            const uint32_t i = b + u * WG;
            if (i < NW) {
              const uint32_t row = mdiv(i, mPW);
              const uint32_t col = i - row * PWc;
              r[u] = src[(size_t(n) * p.L + row_lo + row) * p.RWd + cw0 + col];
            }
          }
#pragma unroll
          for (int u = 0; u < UNR; ++u) {
            const uint32_t i = b + u * WG;
            if (i < NW) {
              const uint32_t row = mdiv(i, mPW);
              tile[row * p.RSw + (i - row * PWc)] = r[u];
            }
          }
        }
      }
    }
    if (lid == 0) tile[p.TW] = sycl::uint4(0u, 0u, 0u, 0u);

    for (uint32_t e = lid; e < OW * CWT; e += WG) {
      const uint32_t w = e / CWT;
      const uint32_t ww = e - w * CWT;
      const uint32_t wp = w + p.PW;
      const uint32_t wq = mdiv(wp, p.mSW);
      const uint32_t wr = wp - wq * p.SW;
      const int ow = int(wq) - int(ww);
      const uint32_t kw = wr + ww * p.SW;
      const bool ok = uint32_t(ow) < uint32_t(p.BW) && kw < uint32_t(p.KW);
      tab[e] = ok ? int(kw * p.sKw + uint32_t(ow) * p.sOw) : int(INV);
    }
    for (uint32_t e = lid; e < HBc * CHT; e += WG) {
      const uint32_t r = e / CHT;
      const uint32_t hh = e - r * CHT;
      const uint32_t hp = h0 + r + p.PH;
      const uint32_t hq = mdiv(hp, p.mSH);
      const uint32_t hr = hp - hq * p.SH;
      const int oh = int(hq) - int(hh);
      const uint32_t kh = hr + hh * p.SH;
      const bool ok = uint32_t(oh) < uint32_t(p.BH) && kh < uint32_t(p.KH);
      tab[OW * CWT + e] = ok ? int(kh * p.sKh + (uint32_t(oh) - ohbase) * p.sOh) : int(INV);
    }
    it.barrier(sycl::access::fence_space::local_space);

    auto lp = sycl::address_space_cast<sycl::access::address_space::local_space,
                                       sycl::access::decorated::no>(
        reinterpret_cast<const U*>(&tile[0]));
    const uint32_t nv = CBc * PRV;
    const uint32_t h0OWV = (h0 * OW) / V;
    sycl::uint4* outv = reinterpret_cast<sycl::uint4*>(p.out);
    for (uint32_t v = lid; v < nv; v += WG) {
      const uint32_t c_l = mdiv(v, mPRV);
      const uint32_t pv = v - c_l * PRV;
      const uint32_t pb = pv * V;
      uint32_t r = mdiv(pb, p.mOW);
      uint32_t w = pb - r * OW;
      const uint32_t cbase = base0 + c_l * p.cstr;
      U bits[V];
#pragma unroll
      for (int j = 0; j < V; ++j) {
        uint32_t ro[CHT];
        uint32_t co[CWT];
#pragma unroll
        for (int hh = 0; hh < CHT; ++hh) ro[hh] = uint32_t(tab[OW * CWT + r * CHT + hh]) + cbase;
#pragma unroll
        for (int ww = 0; ww < CWT; ++ww) co[ww] = uint32_t(tab[w * CWT + ww]);
        T val = static_cast<T>(0);
#pragma unroll
        for (int hh = CHT - 1; hh >= 0; --hh) {
#pragma unroll
          for (int ww = CWT - 1; ww >= 0; --ww) {
            const uint32_t idx = sycl::min(ro[hh] + co[ww], ZOFF);
            val += from_bits(lp[idx]);
          }
        }
        bits[j] = to_bits(val);
        ++w;
        if (w == OW) { w = 0; ++r; }
      }
      uint32_t o[4];
      if constexpr (SZ == 2) {
#pragma unroll
        for (int k = 0; k < 4; ++k) o[k] = uint32_t(bits[2 * k]) | (uint32_t(bits[2 * k + 1]) << 16);
      } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) o[k] = bits[k];
      }
      const size_t ovec = size_t(n * p.C + c0 + c_l) * p.PV + h0OWV + pv;
      outv[ovec] = sycl::uint4(o[0], o[1], o[2], o[3]);
    }
  }
};

// ColKernel for 1-D unit-stride folds of permuted 16-bit inputs
constexpr int kColRows = 64;

struct ColParams {
  const sycl::uint4* in;
  sycl::uint4* out;
  int C, OH, BH, PH, L, RWd, nCB, nHB;
};

template <typename T, int KHT>
struct ColKernel {
  static constexpr int CBt = 16;
  static constexpr int R = kColRows;
  static constexpr int WG = 2 * kColRows;
  static constexpr int PWV = 2 * KHT;
  static constexpr int TR = R + KHT - 1;
  static constexpr int NVEC = TR * PWV;
  static constexpr int UNR = (NVEC + WG - 1) / WG;
  static constexpr int PE = PWV * 8;
  static constexpr int SKV = 1;
  static constexpr int SKE = SKV * 8;
  static constexpr int TV = (TR - 1) * PWV + ((TR - 1) >> 3) * SKV + PWV;

  ColParams p;
  sycl::local_accessor<sycl::uint4, 1> tile;

  static T from_bits(uint16_t u) { T t; t.x = u; return t; }

  template <int P>
  static void gather(T (&val)[8], const uint32_t* lp, uint32_t lb) {
    constexpr int PEh = PE / 2;
    constexpr int SKEh = SKE / 2;
#pragma unroll
    for (int r = 0; r < KHT + 7; ++r) {
      const int lo = sycl::max(0, KHT - 1 - r);
      const int hi = sycl::min(KHT - 1, KHT + 6 - r);
      const int qlo = (lo + P) >> 1;
      const int qhi = (hi + P) >> 1;
      uint32_t w[(KHT + 3) / 2];
#pragma unroll
      for (int q = qlo; q <= qhi; ++q) w[q - qlo] = lp[lb + r * PEh + (r >> 3) * SKEh + q];
#pragma unroll
      for (int kh = hi; kh >= lo; --kh) {
        const int q = (kh + P) >> 1;
        const int h = (kh + P) & 1;
        val[r - (KHT - 1) + kh] += from_bits(uint16_t(w[q - qlo] >> (16 * h)));
      }
    }
  }

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const uint32_t lid = it.get_local_id(0);
    const uint32_t g = it.get_group(0);
    const uint32_t cb = g % p.nCB;
    const uint32_t tq = g / p.nCB;
    const uint32_t hb = tq % p.nHB;
    const uint32_t n = tq / p.nHB;
    const uint32_t c0 = cb * CBt;
    const uint32_t CBc = sycl::min<uint32_t>(CBt, p.C - c0);
    const uint32_t h0 = hb * R;
    const uint32_t HBc = sycl::min<uint32_t>(R, p.OH - h0);
    const uint32_t PWc = (CBc * KHT * 2 + 15) >> 4;
    const int lbase = int(h0) + p.PH - (KHT - 1);

    sycl::uint4 r[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const uint32_t i = lid + u * WG;
      r[u] = sycl::uint4(0u, 0u, 0u, 0u);
      if (i < uint32_t(NVEC)) {
        const uint32_t ri = i / PWV;
        const uint32_t col = i - ri * PWV;
        const int l = lbase + int(ri);
        if (uint32_t(l) < uint32_t(p.BH) && col < PWc) {
          r[u] = p.in[(n * p.L + uint32_t(l)) * p.RWd + cb * PWV + col];
        }
      }
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const uint32_t i = lid + u * WG;
      if (i < uint32_t(NVEC)) {
        const uint32_t ri = i / PWV;
        const uint32_t col = i - ri * PWV;
        tile[ri * PWV + (ri >> 3) * SKV + col] = r[u];
      }
    }
    it.barrier(sycl::access::fence_space::local_space);

    const auto sg = it.get_sub_group();
    const uint32_t sgid = sg.get_group_id()[0];
    const uint32_t lane = sg.get_local_id()[0];
    const uint32_t par = sgid & 1;
    const uint32_t c = (((sgid >> 1) << 1) | (lane >> 3)) * 2 + par;
    const uint32_t hc = lane & 7;
    auto lp = sycl::address_space_cast<sycl::access::address_space::local_space,
                                       sycl::access::decorated::no>(
        reinterpret_cast<const uint32_t*>(&tile[0]));
    const uint32_t lb = (hc * (8 * PE + SKE) + c * KHT - par) >> 1;
    T val[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) val[j] = static_cast<T>(0);
    if (par) { gather<1>(val, lp, lb); }
    else { gather<0>(val, lp, lb); }
    if (c < CBc && hc * 8 < HBc) {
      uint32_t o[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        o[k] = uint32_t(val[2 * k].x) | (uint32_t(val[2 * k + 1].x) << 16);
      }
      const uint32_t ov = (n * p.C + c0 + c) * (uint32_t(p.OH) >> 3) + (h0 >> 3) + hc;
      p.out[ov] = sycl::uint4(o[0], o[1], o[2], o[3]);
    }
  }
};

template <typename T, int KHT>
static void launch_col(const ColParams& p, int64_t tasks) {
  auto& q = at::xpu::getCurrentSYCLQueue();
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<sycl::uint4, 1> tile(sycl::range<1>(ColKernel<T, KHT>::TV), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(tasks * ColKernel<T, KHT>::WG),
                          sycl::range<1>(ColKernel<T, KHT>::WG)),
        ColKernel<T, KHT>{p, tile});
  });
}

template <typename T>
static bool launch_col_kh(int KH, const ColParams& p, int64_t tasks) {
  switch (KH) {
    case 3: launch_col<T, 3>(p, tasks); return true;
    case 5: launch_col<T, 5>(p, tasks); return true;
    case 7: launch_col<T, 7>(p, tasks); return true;
    case 9: launch_col<T, 9>(p, tasks); return true;
    default: return false;
  }
}

struct Plan { bool ok = false; int CB = 0, HB = 0, WG = 0, nCB = 0, nHB = 0, TW = 0; };

inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline int rows_for(int h0, int h1, int KH, int SH, int PH, int BH) {
  const int t = h0 + PH - (KH - 1);
  const int lo = t <= 0 ? 0 : int(cdiv(t, SH));
  const int hi = std::min(BH - 1, (h1 - 1 + PH) / SH);
  return std::max(0, hi - lo + 1);
}

inline Plan make_plan(
    bool B, int C, int KH, int KW, int SH, int PH, int OH, int OW, int BH, int BW, int sz,
    int CHT, int CWT) {
  constexpr int kBudget = 20480;
  const int K2 = KH * KW, L = BH * BW, V = 16 / sz;
  int cbq = 64 / std::gcd(64, OH * OW * sz), hbq = OH;
  if (B) { cbq = 16 / std::gcd(16, K2 * sz); hbq = V / std::gcd(V, OW); }
  std::vector<int> cbs;
  for (int cb = cbq; cb <= C; cb += cbq) cbs.push_back(cb);
  if (C % cbq) cbs.push_back(C);
  std::vector<int> hbs;
  if (B) { for (int hb = hbq; hb <= OH; hb += hbq) hbs.push_back(hb); }
  else { hbs.push_back(OH); }
  struct Cand { double cost; int WG, CB, HB; int64_t tile; };
  std::vector<Cand> cands;
  for (int HB : hbs) {
    const int nHB = int(cdiv(OH, HB));
    int mrows = 1; int64_t rows_tot = 0;
    if (B) {
      mrows = 0;
      for (int i = 0; i < nHB; ++i) {
        const int r = rows_for(i * HB, std::min(OH, (i + 1) * HB), KH, SH, PH, BH);
        mrows = std::max(mrows, r); rows_tot += r;
      }
      rows_tot *= BW;
    }
    for (int CB : cbs) {
      if (B && int64_t(CB) * K2 * sz < 256 && CB < C) continue;
      const int64_t tile = B ? int64_t(mrows) * BW * CB * K2 * sz : int64_t(CB) * K2 * L * sz + 16;
      if (tile > kBudget || tile <= 0) continue;
      const int nCB = int(cdiv(C, CB));
      const double halo = B ? double(rows_tot) / double(std::max(1, L)) : 1.0;
      const int64_t tab = (int64_t(OW) * CWT + int64_t(HB) * CHT) * 4;
      const int64_t smem = tile + 16 + tab;
      if (smem > 24576) continue;
      int cls = 128;
      for (int c : {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128}) {
        if (int64_t(c) * 1024 >= smem) { cls = c; break; }
      }
      const int cbt = C - (nCB - 1) * CB;
      const int hbt = OH - (nHB - 1) * HB;
      for (int WG = 32; WG <= 256; WG += 32) {
        int64_t lane_it = 0;
        const int cbc_v[2] = {CB, cbt}; const int64_t cbn[2] = {nCB - 1, 1};
        const int hbc_v[2] = {HB, hbt}; const int64_t hbn[2] = {nHB - 1, 1};
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            const int64_t nv = int64_t(cbc_v[a]) * hbc_v[b] * OW / V;
            lane_it += cbn[a] * hbn[b] * cdiv(nv, WG) * WG;
          }
        }
        const double alu = double(lane_it) / double(int64_t(C) * OH * OW / V);
        const int occ = std::min(64, (128 / cls) * (WG / 32));
        cands.push_back({halo * alu * std::max(1.0, 16.0 / occ), WG, CB, HB, tile});
      }
    }
  }
  Plan plan;
  if (cands.empty()) return plan;
  double cmin = cands[0].cost;
  for (auto& c : cands) cmin = std::min(cmin, c.cost);
  const Cand* best = nullptr;
  for (auto& c : cands) {
    if (c.cost > cmin * 1.03) continue;
    if (!best || c.WG > best->WG || (c.WG == best->WG && c.tile > best->tile)) best = &c;
  }
  plan.ok = true; plan.CB = best->CB; plan.HB = best->HB; plan.WG = best->WG;
  plan.nCB = int(cdiv(C, plan.CB)); plan.nHB = int(cdiv(OH, plan.HB));
  if (B) {
    int mrows = 0;
    for (int i = 0; i < plan.nHB; ++i)
      mrows = std::max(mrows, rows_for(i * plan.HB, std::min(OH, (i + 1) * plan.HB), KH, SH, PH, BH));
    plan.TW = int(int64_t(mrows) * BW * (int64_t(plan.CB) * K2 * sz / 16));
  } else {
    plan.TW = int((int64_t(plan.CB) * K2 * L * sz + 30) / 16 + 1);
  }
  if (plan.TW < 1) plan.TW = 1;
  return plan;
}

template <typename T, int CHT, int CWT>
static void launch_tiles(const Params& p, int64_t tasks) {
  auto& q = at::xpu::getCurrentSYCLQueue();
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<sycl::uint4, 1> tile(sycl::range<1>(p.TW + 1), cgh);
    sycl::local_accessor<int, 1> tab(sycl::range<1>(p.OW * CWT + p.HB * CHT), cgh);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(tasks * p.WG), sycl::range<1>(p.WG)),
        TileKernel<T, CHT, CWT>{p, tile, tab});
  });
}

template <typename T>
static void launch_class(int cls, const Params& p, int64_t tasks) {
  switch (cls) {
    case 0: launch_tiles<T, 1, 1>(p, tasks); break;
    case 1: launch_tiles<T, 2, 2>(p, tasks); break;
    case 2: launch_tiles<T, 9, 1>(p, tasks); break;
    default: launch_tiles<T, 4, 4>(p, tasks); break;
  }
}

static bool try_fast_path(
    const Tensor& input_, const Tensor& output, int64_t OH, int64_t OW, int64_t KH, int64_t KW,
    int64_t DH, int64_t DW, int64_t PH, int64_t PW, int64_t SH, int64_t SW) {
  if (DH != 1 || DW != 1) return false;
  const auto st = input_.scalar_type();
  if (st != at::kHalf && st != at::kBFloat16 && st != at::kFloat) return false;
  if (output.scalar_type() != st || !output.is_contiguous()) return false;
  if (input_.dim() != 2 && input_.dim() != 3) return false;
  const Tensor in3 = input_.dim() == 2 ? input_.unsqueeze(0) : input_;
  const int64_t N = in3.size(0), CK = in3.size(1), L = in3.size(2);
  const int64_t K2 = KH * KW;
  if (N <= 0 || CK <= 0 || L <= 0 || K2 <= 0 || OH <= 0 || OW <= 0) return false;
  if (CK % K2 != 0) return false;
  const int64_t C = CK / K2;
  const int64_t BH = (OH + 2 * PH - (KH - 1) - 1) / SH + 1;
  const int64_t BW = (OW + 2 * PW - (KW - 1) - 1) / SW + 1;
  if (BH * BW != L) return false;
  if (output.numel() != N * C * OH * OW) return false;
  if (in3.numel() >= (int64_t(1) << 31) || output.numel() >= (int64_t(1) << 31)) return false;
  const int sz = int(input_.element_size());
  const int V = 16 / sz;
  if ((OH * OW) % V != 0) return false;
  if (reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 != 0) return false;
  bool layoutB;
  if (in3.is_contiguous()) { layoutB = false; }
  else if (in3.transpose(1, 2).is_contiguous()) {
    layoutB = true;
    if ((CK * sz) % 16 != 0 || reinterpret_cast<uintptr_t>(in3.const_data_ptr()) % 16 != 0) return false;
  } else { return false; }
  if (layoutB && sz == 2 && KW == 1 && SW == 1 && PW == 0 && OW == 1 && SH == 1 && BW == 1 &&
      OH % 8 == 0 && (KH == 3 || KH == 5 || KH == 7 || KH == 9)) {
    ColParams cp{};
    cp.in = static_cast<const sycl::uint4*>(in3.const_data_ptr());
    cp.out = static_cast<sycl::uint4*>(output.data_ptr());
    cp.C = int(C); cp.OH = int(OH); cp.BH = int(BH); cp.PH = int(PH); cp.L = int(L);
    cp.RWd = int(CK * sz / 16);
    cp.nCB = int(cdiv(C, 16)); cp.nHB = int(cdiv(OH, kColRows));
    const int64_t tasks = N * cp.nCB * cp.nHB;
    if (st == at::kHalf) return launch_col_kh<c10::Half>(int(KH), cp, tasks);
    return launch_col_kh<c10::BFloat16>(int(KH), cp, tasks);
  }
  const int CHa = int((KH - 1) / SH + 1), CWa = int((KW - 1) / SW + 1);
  int cls;
  if (CHa <= 1 && CWa <= 1) cls = 0;
  else if (CHa <= 2 && CWa <= 2) cls = 1;
  else if (CHa <= 9 && CWa <= 1) cls = 2;
  else if (CHa <= 4 && CWa <= 4) cls = 3;
  else return false;
  const int CHT = cls == 0 ? 1 : cls == 1 ? 2 : cls == 2 ? 9 : 4;
  const int CWT = cls == 0 ? 1 : cls == 1 ? 2 : cls == 2 ? 1 : 4;

  using Key = std::array<int64_t, 14>;
  static std::map<Key, Plan> cache;
  const Key key{layoutB, C, KH, KW, SH, PH, OH, OW, BH, BW, sz, CHT, CWT, 0};
  auto it = cache.find(key);
  if (it == cache.end()) {
    it = cache.emplace(key, make_plan(layoutB, int(C), int(KH), int(KW), int(SH), int(PH), int(OH),
                                      int(OW), int(BH), int(BW), sz, CHT, CWT)).first;
  }
  const Plan& pl = it->second;
  if (!pl.ok) return false;

  Params p{};
  p.in = static_cast<const char*>(in3.const_data_ptr());
  p.out = static_cast<char*>(output.data_ptr());
  p.N = int(N); p.C = int(C); p.OH = int(OH); p.OW = int(OW); p.BH = int(BH); p.BW = int(BW);
  p.KH = int(KH); p.KW = int(KW); p.SH = int(SH); p.SW = int(SW); p.PH = int(PH); p.PW = int(PW);
  p.L = int(L); p.K2 = int(K2); p.CK = int(CK);
  p.layoutB = layoutB ? 1 : 0;
  p.CB = pl.CB; p.HB = pl.HB; p.nCB = pl.nCB; p.nHB = pl.nHB; p.WG = pl.WG; p.TW = pl.TW;
  const int EPW = 16 / sz;
  if (layoutB) {
    p.RSw = int(int64_t(pl.CB) * K2 * sz / 16); p.RWd = int(CK * sz / 16);
    const int RSe = p.RSw * EPW;
    p.sKh = int(KW); p.sKw = 1; p.sOh = int(BW) * RSe; p.sOw = RSe; p.cstr = int(K2);
  } else {
    p.RSw = 1; p.RWd = 1;
    p.sKh = int(KW * L); p.sKw = int(L); p.sOh = int(BW); p.sOw = 1; p.cstr = int(K2 * L);
  }
  const int cbt = int(C) - (pl.nCB - 1) * pl.CB;
  const int hbt = int(OH) - (pl.nHB - 1) * pl.HB;
  p.PWfull = int(cdiv(int64_t(pl.CB) * K2 * sz, 16));
  p.PWtail = int(cdiv(int64_t(cbt) * K2 * sz, 16));
  p.PRVfull = int(int64_t(pl.HB) * OW / V);
  p.PRVtail = int(int64_t(hbt) * OW / V);
  p.PV = int(OH * OW / V);
  p.mOW = make_magic(uint32_t(OW));
  p.mSH = make_magic(uint32_t(SH)); p.mSW = make_magic(uint32_t(SW));
  p.mPRVfull = make_magic(uint32_t(p.PRVfull)); p.mPRVtail = make_magic(uint32_t(p.PRVtail));
  p.mPWfull = make_magic(uint32_t(p.PWfull)); p.mPWtail = make_magic(uint32_t(p.PWtail));

  const int64_t tasks = N * pl.nCB * pl.nHB;
  if (st == at::kHalf) launch_class<c10::Half>(cls, p, tasks);
  else if (st == at::kBFloat16) launch_class<c10::BFloat16>(cls, p, tasks);
  else launch_class<float>(cls, p, tasks);
  return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void col2im_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  TORCH_CHECK(output_size.size() == 2, "It is expected output_size equals to 2, but got size ", output_size.size());
  TORCH_CHECK(kernel_size.size() == 2, "It is expected kernel_size equals to 2, but got size ", kernel_size.size());
  TORCH_CHECK(dilation.size() == 2, "It is expected dilation equals to 2, but got size ", dilation.size());
  TORCH_CHECK(padding.size() == 2, "It is expected padding equals to 2, but got size ", padding.size());
  TORCH_CHECK(stride.size() == 2, "It is expected stride equals to 2, but got size ", stride.size());

  auto output_height = output_size[0];
  auto output_width = output_size[1];
  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto dilation_height = dilation[0];
  auto dilation_width = dilation[1];
  auto pad_height = padding[0];
  auto pad_width = padding[1];
  auto stride_height = stride[0];
  auto stride_width = stride[1];

  col2im_shape_check(input_, Tensor(), output_height, output_width, kernel_height, kernel_width,
                     dilation_height, dilation_width, pad_height, pad_width, stride_height, stride_width);

  bool batched_input = true;
  Tensor input = input_;
  if (input.dim() == 2) {
    batched_input = false;
    input = input.view({1, input.size(0), input.size(1)});
  }

  auto batch_size = input.size(0);
  auto n_input_plane = input.size(1);
  auto n_output_plane = n_input_plane / (kernel_width * kernel_height);

  output.resize_({batch_size, n_output_plane, output_height, output_width});

  // Try fast path (single launch, SLM-staged, no contiguous() copy)
  if (try_fast_path(input, output, output_height, output_width,
                    kernel_height, kernel_width, dilation_height, dilation_width,
                    pad_height, pad_width, stride_height, stride_width)) {
    if (!batched_input) output.resize_({n_output_plane, output_height, output_width});
    return;
  }

  // Fallback: baseline path (per-batch launch with contiguous() copy)
  Tensor input_contig = input.contiguous();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
      at::ScalarType::BFloat16, at::ScalarType::Half, at::ScalarType::Bool,
      input.scalar_type(), "col2im_xpu", [&] {
        auto height_col = (output_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
        auto width_col = (output_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;
        for (int64_t elt = 0; elt < batch_size; elt++) {
          Tensor input_n = input_contig.select(0, elt);
          Tensor output_n = output.select(0, elt);
          col2im_launch_baseline<scalar_t>(
              input_n.const_data_ptr<scalar_t>(), n_output_plane, output_height, output_width,
              height_col, width_col, kernel_height, kernel_width, pad_height, pad_width,
              stride_height, stride_width, dilation_height, dilation_width,
              output_n.data_ptr<scalar_t>());
        }
        if (!batched_input) output.resize_({n_output_plane, output_height, output_width});
      });
}

} // namespace at::native::xpu
