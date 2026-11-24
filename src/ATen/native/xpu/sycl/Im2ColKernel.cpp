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

#include <ATen/native/xpu/sycl/Im2ColKernel.h>

#include <algorithm>
#include <cmath>

namespace at::native::xpu {

// ---------------------------------------------------------------------------
// Baseline: one work-item per output element, per-batch launch (unchanged)
// ---------------------------------------------------------------------------

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void im2col_kernel_func(
    const T* in_data,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t height_col,
    const int64_t width_col,
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
  if (id >= channels * height_col * width_col) {
    return;
  }

  int64_t w_out = id % width_col;
  id /= width_col;

  int64_t h_out = id % height_col;
  int64_t channel_in = id / height_col;
  int64_t channel_out = channel_in * kernel_h * kernel_w;
  int64_t h_in = h_out * stride_h - pad_h;
  int64_t w_in = w_out * stride_w - pad_w;

  out_ptr += (channel_out * height_col + h_out) * width_col + w_out;
  in_ptr += (channel_in * height + h_in) * width + w_in;

  for (int64_t i = 0; i < kernel_h; ++i) {
    for (int64_t j = 0; j < kernel_w; ++j) {
      int64_t h = h_in + i * dilation_h;
      int64_t w = w_in + j * dilation_w;
      *out_ptr = (h >= 0 && w >= 0 && h < height && w < width)
          ? in_ptr[i * dilation_h * width + j * dilation_w]
          : static_cast<T>(0);
      out_ptr += height_col * width_col;
    }
  }
}

template <typename T>
static void im2col_launch_baseline(
    const T* data_im,
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
    T* data_col) {
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  auto total_threads = channels * output_width * output_height;

  auto in_data = data_im;
  auto out_data = data_col;

  int64_t local_range =
      at::xpu::getKernelMaxWorkGroupSize<im2col_kernel_func<T>>();
  int64_t global_range =
      ((total_threads + local_range - 1) / local_range) * local_range;

  sycl_kernel_submit<im2col_kernel_func<T>>(
      global_range,
      local_range,
      sycl_queue,
      0,
      in_data,
      channels,
      height,
      width,
      height_col,
      width_col,
      kernel_h,
      kernel_w,
      pad_h,
      pad_w,
      stride_h,
      stride_w,
      dilation_h,
      dilation_w,
      out_data);
}

// ---------------------------------------------------------------------------
// Candidate: one launch per call, persistent work-groups over SLM channel-block tiles
// ---------------------------------------------------------------------------

namespace {

template <int ESZ>
struct Bits;
template <> struct Bits<2> { using type = uint16_t; };
template <> struct Bits<4> { using type = uint32_t; };

enum Loader : int { kChannelsLastVec = 0, kPlaneVec = 1, kGeneric = 2 };

struct Params {
  int64_t sn, sc, sh, sw;
  int spix;
  int C, H, W, plane, planepad;
  int L, OW, KW, K, KL;
  int pad_h, pad_w, str_h, str_w, dil_h, dil_w;
  int CB, nblk, T;
  int RS, P, gshift;
  int PIX, CH;
  int dq, dr;
  int pa, pb;
  int NL;
};

template <int ESZ, int LOADER, int MAXLD, bool CVEC>
struct TileKernel {
  using U = typename Bits<ESZ>::type;
  static constexpr int V = 16 / ESZ;
  static constexpr int LOGV = V == 8 ? 3 : 2;
  typedef U VecU __attribute__((ext_vector_type(V)));
  typedef uint16_t VecT __attribute__((ext_vector_type(V)));
  typedef uint32_t VecW __attribute__((ext_vector_type(4)));

  struct Tile { int n, c0, cbv; };

  Tile decode(int t) const {
    Tile r;
    r.n = t / p.nblk;
    r.c0 = (t - r.n * p.nblk) * p.CB;
    r.cbv = sycl::min(p.CB, p.C - r.c0);
    return r;
  }

  void issue(const Tile& tt, int lid, int WG, VecW (&v)[MAXLD], int (&so)[MAXLD]) const {
    const U* src = in + static_cast<int64_t>(tt.n) * p.sn + static_cast<int64_t>(tt.c0) * p.sc;
    if constexpr (LOADER == kChannelsLastVec) {
      const int NV = tt.cbv / V;
      const int total = p.plane * NV;
      const int a = WG / NV, b = WG - a * NV;
      int pp = lid / NV;
      int cq = lid - pp * NV;
#pragma unroll
      for (int j = 0; j < MAXLD; ++j) {
        const int msk = -static_cast<int>(lid + j * WG < total);
        so[j] = ((pp * p.CB + cq * V) * ESZ) | ~msk;
        v[j] = *reinterpret_cast<const VecW*>(src + ((pp * p.spix + cq * V) & msk));
        cq += b; pp += a;
        const int c = -static_cast<int>(cq >= NV);
        cq -= NV & c; pp -= c;
      }
    } else {
      const int total = tt.cbv * (p.plane / V);
      const int sc = static_cast<int>(p.sc);
      const int m = lid * V;
      int cc = m / p.plane;
      int pp = m - cc * p.plane;
#pragma unroll
      for (int j = 0; j < MAXLD; ++j) {
        const int msk = -static_cast<int>(lid + j * WG < total);
        so[j] = ((cc * p.planepad + pp) * ESZ) | ~msk;
        v[j] = *reinterpret_cast<const VecW*>(src + ((cc * sc + pp) & msk));
        pp += p.pb; cc += p.pa;
        const int c = -static_cast<int>(pp >= p.plane);
        pp -= p.plane & c; cc -= c;
      }
    }
  }

  void commit(const Tile& tt, int lid, int WG, char* sd, const VecW (&v)[MAXLD],
              const int (&so)[MAXLD]) const {
    U* data = reinterpret_cast<U*>(sd);
    const U* src = in + static_cast<int64_t>(tt.n) * p.sn + static_cast<int64_t>(tt.c0) * p.sc;
    if constexpr (LOADER == kGeneric) {
      const int total = tt.cbv * p.plane;
      for (int m = lid; m < total; m += WG) {
        const int c = m / p.plane;
        const int pq = m - c * p.plane;
        const int h = pq / p.W;
        const int w = pq - h * p.W;
        data[c * p.planepad + pq] = src[c * p.sc + h * p.sh + w * p.sw];
      }
    } else {
#pragma unroll
      for (int j = 0; j < MAXLD; ++j)
        if (so[j] >= 0)
          *reinterpret_cast<VecW*>(sd + so[j]) = v[j];
    }
  }

  void gather(const Tile& tt, int lid, int WG, const char* sd, const uint16_t* tab) const {
    const int KL = p.KL;
    const int64_t A = (static_cast<int64_t>(tt.n) * p.C + tt.c0) * KL;
    const int64_t E = A + static_cast<int64_t>(tt.cbv) * KL;
    const int lead = static_cast<int>(A & (V - 1));
    const int64_t gA = A - lead;
    const int ng = static_cast<int>(((E + V - 1) >> LOGV) - (A >> LOGV));
    const int klo = lead ? 1 : 0;
    const int tail = static_cast<int>(E & (V - 1));
    const int khi = ng - (tail ? 1 : 0);
    U* outp = out + gA;
    int k = klo + lid;
    const int off = k * V - lead;
    int crel = off / KL;
    int r = off - crel * KL;
    for (; k < khi; k += WG) {
      const int s = (-r) & (V - 1);
      const VecT tv = *reinterpret_cast<const VecT*>(tab + (s >> p.gshift) * p.RS + r + s);
      const char* d = sd + crel * p.CH;
      VecU o;
#pragma unroll
      for (int i = 0; i < V; ++i)
        o[i] = *reinterpret_cast<const U*>(d + tv[i]);
      *reinterpret_cast<VecU*>(outp + k * V) = o;
      r += p.dr; crel += p.dq;
      if (r >= KL) { r -= KL; ++crel; }
    }
    const bool first = lead && lid == 0;
    const bool last = tail && lid == WG - 1;
    if (first || last) {
      const int f0 = first ? 0 : static_cast<int>((E - tail) - A);
      const int f1 = first ? V - lead : static_cast<int>(E - A);
      for (int f = f0; f < f1; ++f) {
        const int c = f / KL;
        const int rr = f - c * KL;
        out[A + f] = *reinterpret_cast<const U*>(sd + c * p.CH + tab[rr]);
      }
    }
  }

  void gather_cvec(const Tile& tt, int lid, int WG, const char* sd, const uint16_t* tab) const {
    const int KL = p.KL;
    const int nx = KL / V;
    const int nb = (tt.cbv / V) * nx;
    U* outt = out + (static_cast<int64_t>(tt.n) * p.C + tt.c0) * KL;
    for (int b = lid; b < nb; b += WG) {
      const int cv = b / nx;
      const int xg = b - cv * nx;
      const VecT tv = *reinterpret_cast<const VecT*>(tab + xg * V);
      const char* d = sd + cv * 16;
      VecU src[V];
#pragma unroll
      for (int i = 0; i < V; ++i)
        src[i] = *reinterpret_cast<const VecU*>(d + tv[i]);
      U* o = outt + (cv * V) * KL + xg * V;
#pragma unroll
      for (int c = 0; c < V; ++c) {
        VecU r;
#pragma unroll
        for (int i = 0; i < V; ++i)
          r[i] = src[i][c];
        *reinterpret_cast<VecU*>(o + c * KL) = r;
      }
    }
  }

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
    const int lid = static_cast<int>(it.get_local_id(0));
    const int WG = static_cast<int>(it.get_local_range(0));
    const int G = static_cast<int>(it.get_group_range(0));
    char* sd = reinterpret_cast<char*>(
        slm_data.template get_multi_ptr<sycl::access::decorated::no>().get());
    uint16_t* tab = reinterpret_cast<uint16_t*>(
        slm_tab.template get_multi_ptr<sycl::access::decorated::no>().get());

    int t = static_cast<int>(it.get_group(0));
    VecW v[MAXLD];
    int so[MAXLD];
    Tile cur = decode(t);
    if constexpr (LOADER != kGeneric)
      issue(cur, lid, WG, v, so);

    U* data = reinterpret_cast<U*>(sd);
    if constexpr (LOADER == kChannelsLastVec) {
      for (int c = lid; c < p.CB; c += WG)
        data[p.plane * p.CB + c] = U(0);
    } else {
      for (int c = lid; c < p.CB; c += WG)
        data[c * p.planepad + p.plane] = U(0);
    }
    for (int l = lid; l < p.L; l += WG) {
      const int oh = l / p.OW;
      const int ow = l - oh * p.OW;
      const int hb = oh * p.str_h - p.pad_h;
      const int wb = ow * p.str_w - p.pad_w;
      int i = 0, j = 0;
      for (int kk = 0; kk < p.K; ++kk) {
        const int h = hb + i * p.dil_h;
        const int w = wb + j * p.dil_w;
        const int e = (static_cast<unsigned>(h) < static_cast<unsigned>(p.H) &&
                       static_cast<unsigned>(w) < static_cast<unsigned>(p.W))
            ? (h * p.W + w) * p.PIX
            : p.plane * p.PIX;
        const int r = kk * p.L + l;
        for (int row = 0; row < p.P; ++row) {
          const int s = row << p.gshift;
          uint16_t* tr = tab + row * p.RS;
          tr[r + s] = static_cast<uint16_t>(e);
          if (r + s + p.KL < p.RS)
            tr[r + s + p.KL] = static_cast<uint16_t>(e + p.CH);
          if (r < s)
            tr[r] = static_cast<uint16_t>(p.plane * p.PIX);
        }
        if (++j == p.KW) { j = 0; ++i; }
      }
    }

    for (;;) {
      sycl::group_barrier(it.get_group());
      commit(cur, lid, WG, sd, v, so);
      sycl::group_barrier(it.get_group());
      const int nt = t + G;
      const Tile nxt = decode(nt < p.T ? nt : t);
      if constexpr (LOADER != kGeneric)
        if (nt < p.T)
          issue(nxt, lid, WG, v, so);
      if constexpr (CVEC)
        gather_cvec(cur, lid, WG, sd, tab);
      else
        gather(cur, lid, WG, sd, tab);
      if (nt >= p.T) break;
      t = nt; cur = nxt;
    }
  }

  TileKernel(const U* in_, U* out_, Params p_, sycl::local_accessor<VecU, 1> d_,
             sycl::local_accessor<VecT, 1> t_)
      : in(in_), out(out_), p(p_), slm_data(d_), slm_tab(t_) {}

 private:
  const U* in;
  U* out;
  Params p;
  sycl::local_accessor<VecU, 1> slm_data;
  sycl::local_accessor<VecT, 1> slm_tab;
};

static inline int64_t round_up(int64_t a, int64_t b) { return (a + b - 1) / b * b; }

static inline int xe_cores() {
  static const int v = [] {
    auto dev = at::xpu::getCurrentSYCLQueue().get_device();
    int cu = static_cast<int>(dev.get_info<sycl::info::device::max_compute_units>());
    return std::max(1, cu / 8);
  }();
  return v;
}

template <int ESZ, int LOADER, int MAXLD, bool CVEC>
static void launch(const void* in, void* out, const Params& p, int64_t groups, int WG,
                   int64_t data_bytes, int64_t tab_elems) {
  using K = TileKernel<ESZ, LOADER, MAXLD, CVEC>;
  using U = typename K::U;
  using VecU = typename K::VecU;
  using VecT = typename K::VecT;
  constexpr int V = K::V;
  auto& q = at::xpu::getCurrentSYCLQueue();
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<VecU, 1> d(sycl::range<1>((data_bytes + 15) / 16), cgh);
    sycl::local_accessor<VecT, 1> t(sycl::range<1>(tab_elems / V), cgh);
    cgh.parallel_for<K>(
        sycl::nd_range<1>(sycl::range<1>(groups * WG), sycl::range<1>(WG)),
        K(static_cast<const U*>(in), static_cast<U*>(out), p, d, t));
  });
}

template <int ESZ, int LOADER, int MAXLD, bool CVEC>
static int64_t max_wg() {
  return at::xpu::getKernelMaxWorkGroupSize<TileKernel<ESZ, LOADER, MAXLD, CVEC>>();
}

static bool try_fast_path(const at::Tensor& input, int64_t N, int64_t sn, int64_t C,
                          int64_t sc, int64_t H, int64_t sh, int64_t W, int64_t sw,
                          int64_t KH, int64_t KW, int64_t dil_h, int64_t dil_w,
                          int64_t pad_h, int64_t pad_w, int64_t str_h, int64_t str_w,
                          int64_t OH, int64_t OW, at::Tensor& out) {
  const int esz = static_cast<int>(input.element_size());
  if (esz != 2 && esz != 4) return false;
  const int64_t V = 16 / esz;
  const int64_t plane = H * W;
  const int64_t L = OH * OW;
  const int64_t K = KH * KW;
  const int64_t KL = K * L;
  const uintptr_t ip = reinterpret_cast<uintptr_t>(input.const_data_ptr());
  const uintptr_t op = reinterpret_cast<uintptr_t>(out.data_ptr());
  if (op % 16 != 0 || KL < 3 * V || C * KL >= (int64_t(1) << 31)) return false;
  if (std::abs(sc) * C + std::abs(sh) * H + std::abs(sw) * W >= (int64_t(1) << 31)) return false;

  int loader = kGeneric;
  int64_t spix = 0;
  const bool al16 = ip % 16 == 0 && (N == 1 || (sn * esz) % 16 == 0);
  if (C > 1 && sc == 1 && C % V == 0 && al16) {
    spix = W > 1 ? sw : (H > 1 ? sh : C);
    if ((H == 1 || sh == W * spix) && (spix * esz) % 16 == 0)
      loader = kChannelsLastVec;
  }
  if (loader == kGeneric && (W == 1 || sw == 1) && (H == 1 || sh == W) && plane % V == 0 &&
      al16 && (C == 1 || (sc * esz) % 16 == 0))
    loader = kPlaneVec;

  const bool pixmajor = loader == kChannelsLastVec;
  const int64_t planepad = round_up(plane + 1, V);
  int64_t g = 1;
  while (g < V && KL % (2 * g) == 0) g *= 2;
  const int64_t P = V / g;
  int gshift = 0;
  while ((int64_t(1) << gshift) < g) ++gshift;
  const int64_t RS = round_up(KL + 2 * V, V);
  const int64_t tab_bytes = P * RS * 2;
  const int64_t unit = pixmajor ? V : 1;
  const int64_t ch_bytes = (pixmajor ? plane + 1 : planepad) * esz;

  int64_t budget = 32768, WG = 256, wgs_per_core = 4;
  int64_t cbmax = (budget - tab_bytes) / ch_bytes / unit * unit;
  if (cbmax < unit) {
    budget = 65536; WG = 512; wgs_per_core = 2;
    cbmax = (budget - tab_bytes) / ch_bytes / unit * unit;
  }
  if (cbmax < unit) return false;
  cbmax = std::min(cbmax, round_up(C, unit));

  const int64_t conc = static_cast<int64_t>(xe_cores()) * wgs_per_core;
  const double ovh = 0.68e6 / static_cast<double>(conc) / static_cast<double>((plane + KL) * esz);
  const int64_t nmin = (C + cbmax - 1) / cbmax;
  const int64_t nmax = std::min(4 * nmin, (C + unit - 1) / unit);
  int64_t best_cb = cbmax, best_nb = nmin;
  double best_cost = 1e300;
  for (int64_t nb = nmin; nb <= nmax; ++nb) {
    const int64_t cb = round_up((C + nb - 1) / nb, unit);
    const int64_t nbe = (C + cb - 1) / cb;
    const double cost =
        static_cast<double>((N * nbe + conc - 1) / conc) * (static_cast<double>(cb) + ovh);
    if (cost < best_cost) { best_cost = cost; best_cb = cb; best_nb = nbe; }
  }
  const int64_t T = N * best_nb;
  if (T >= (int64_t(1) << 31)) return false;

  const int64_t PIX = pixmajor ? best_cb * esz : esz;
  const int64_t CH = pixmajor ? esz : planepad * esz;
  const int64_t data_bytes = pixmajor ? (plane + 1) * best_cb * esz : round_up(best_cb * planepad, V) * esz;
  if (plane * PIX + (best_cb > 1 ? CH : 0) > 65535) return false;

  Params p;
  p.sn = sn; p.sc = sc; p.sh = sh; p.sw = sw; p.spix = int(spix);
  p.C = int(C); p.H = int(H); p.W = int(W); p.plane = int(plane); p.planepad = int(planepad);
  p.L = int(L); p.OW = int(OW); p.KW = int(KW); p.K = int(K); p.KL = int(KL);
  p.pad_h = int(pad_h); p.pad_w = int(pad_w); p.str_h = int(str_h); p.str_w = int(str_w);
  p.dil_h = int(dil_h); p.dil_w = int(dil_w);
  p.CB = int(best_cb); p.nblk = int(best_nb); p.T = int(T);
  p.RS = int(RS); p.P = int(P); p.gshift = gshift;
  p.PIX = int(PIX); p.CH = int(CH);
  p.dq = int((WG * V) / KL); p.dr = int((WG * V) % KL);
  p.pa = int((WG * V) / plane); p.pb = int((WG * V) % plane);
  p.NL = int((best_cb * plane / V + WG - 1) / WG);
  if (p.NL > 8) return false;
  const int64_t tab_elems = P * RS;
  const int64_t groups = std::min(T, conc);

  const void* in = input.const_data_ptr();
  void* o = out.data_ptr();
#define KB_IM2COL_LAUNCH(E, LD, M, CV)                                    \
  {                                                                       \
    if (max_wg<E, LD, M, CV>() < WG)                                      \
      return false;                                                       \
    launch<E, LD, M, CV>(in, o, p, groups, int(WG), data_bytes, tab_elems); \
    return true;                                                          \
  }
  const bool small = p.NL <= 4;
  const bool cvec = pixmajor && KL % V == 0;
  if (esz == 2) {
    if (loader == kChannelsLastVec) {
      if (cvec && small) KB_IM2COL_LAUNCH(2, kChannelsLastVec, 4, true)
      if (cvec) KB_IM2COL_LAUNCH(2, kChannelsLastVec, 8, true)
      if (small) KB_IM2COL_LAUNCH(2, kChannelsLastVec, 4, false)
      KB_IM2COL_LAUNCH(2, kChannelsLastVec, 8, false)
    }
    if (loader == kPlaneVec) {
      if (small) KB_IM2COL_LAUNCH(2, kPlaneVec, 4, false)
      KB_IM2COL_LAUNCH(2, kPlaneVec, 8, false)
    }
    KB_IM2COL_LAUNCH(2, kGeneric, 1, false)
  } else {
    if (loader == kChannelsLastVec) {
      if (cvec && small) KB_IM2COL_LAUNCH(4, kChannelsLastVec, 4, true)
      if (cvec) KB_IM2COL_LAUNCH(4, kChannelsLastVec, 8, true)
      if (small) KB_IM2COL_LAUNCH(4, kChannelsLastVec, 4, false)
      KB_IM2COL_LAUNCH(4, kChannelsLastVec, 8, false)
    }
    if (loader == kPlaneVec) {
      if (small) KB_IM2COL_LAUNCH(4, kPlaneVec, 4, false)
      KB_IM2COL_LAUNCH(4, kPlaneVec, 8, false)
    }
    KB_IM2COL_LAUNCH(4, kGeneric, 1, false)
  }
#undef KB_IM2COL_LAUNCH
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void im2col_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  TORCH_CHECK(kernel_size.size() == 2, "It is expected kernel_size equals to 2, but got size ", kernel_size.size());
  TORCH_CHECK(dilation.size() == 2, "It is expected dilation equals to 2, but got size ", dilation.size());
  TORCH_CHECK(padding.size() == 2, "It is expected padding equals to 2, but got size ", padding.size());
  TORCH_CHECK(stride.size() == 2, "It is expected stride equals to 2, but got size ", stride.size());

  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto dilation_height = dilation[0];
  auto dilation_width = dilation[1];
  auto pad_height = padding[0];
  auto pad_width = padding[1];
  auto stride_height = stride[0];
  auto stride_width = stride[1];

  im2col_shape_check(input_, Tensor(), kernel_height, kernel_width, dilation_height,
                     dilation_width, pad_height, pad_width, stride_height, stride_width);

  bool batched_input = true;
  Tensor input = input_;
  if (input.dim() == 3) {
    batched_input = false;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
  }

  auto batch_size = input.size(0);
  auto n_input_plane = input.size(1);
  auto input_height = input.size(2);
  auto input_width = input.size(3);

  auto output_height = (input_height + 2 * pad_height -
                        (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
  auto output_width = (input_width + 2 * pad_width -
                       (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;
  auto n_output_plane = n_input_plane * kernel_width * kernel_height;
  auto output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});

  // Try fast path (single launch, SLM-staged, no contiguous() copy)
  const int64_t N = batch_size;
  const int64_t sn = input.stride(0);
  const int64_t C = n_input_plane, H = input_height, W = input_width;
  const int64_t sc = input.stride(1), sh = input.stride(2), sw = input.stride(3);
  if (try_fast_path(input, N, sn, C, sc, H, sh, W, sw,
                    kernel_height, kernel_width, dilation_height, dilation_width,
                    pad_height, pad_width, stride_height, stride_width,
                    output_height, output_width, output)) {
    if (!batched_input)
      output.resize_({n_output_plane, output_length});
    return;
  }

  // Fallback: baseline path (per-batch launch with contiguous() copy)
  Tensor input_contig = input.contiguous();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
      kHalf, kBFloat16, kBool, input.scalar_type(), "im2col_xpu", [&] {
        Tensor input_n;
        Tensor output_n;
        for (int64_t elt = 0; elt < batch_size; elt++) {
          input_n = input_contig.select(0, elt);
          output_n = output.select(0, elt);
          im2col_launch_baseline<scalar_t>(
              input_n.const_data_ptr<scalar_t>(), n_input_plane, input_height, input_width,
              output_height, output_width, kernel_height, kernel_width, pad_height, pad_width,
              stride_height, stride_width, dilation_height, dilation_width,
              output_n.data_ptr<scalar_t>());
        }
        if (!batched_input)
          output.resize_({n_output_plane, output_length});
      });
}

} // namespace at::native::xpu
