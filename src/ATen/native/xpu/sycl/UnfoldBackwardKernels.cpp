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
#include <ATen/core/Tensor.h>
#include <ATen/native/UnfoldBackward.h>
#include <c10/core/ScalarType.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/UnfoldBackwardKernels.h>

#include <cstring>
#include <numeric>

namespace at::native::xpu {

// Branchless unsigned division by a runtime constant; valid for n <= INT32_MAX.
struct Div32 {
  uint32_t m = 1;
  uint32_t s = 0;
  Div32() = default;
  explicit Div32(uint32_t d) {
    uint32_t sh = 0;
    while (sh < 31 && (uint32_t(1) << sh) < d) ++sh;
    s = sh;
    uint64_t one = 1;
    m = static_cast<uint32_t>(((one << 32) * ((one << sh) - d)) / d + 1);
  }
  inline uint32_t div(uint32_t n) const { return (sycl::mul_hi(n, m) + n) >> s; }
};

constexpr int kBlockWG = 256;
constexpr int kSlmU4 = 512;  // 8 KB of SLM per work-group
constexpr int kSlmBytes = kSlmU4 * 16;

template <typename scalar_t>
struct UnfoldBwdBlockKernel {
  static constexpr int VO = 16 / sizeof(scalar_t);  // elements per 16 B unit
  static constexpr int LPL = kSlmU4 / kBlockWG;     // 16 B loads per work-item

  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::nd_item<1> it) const {
    const int lid = static_cast<int>(it.get_local_id(0));
    const int64_t o0 = static_cast<int64_t>(it.get_group(0)) * nb_;
    const int nbk = static_cast<int>(sycl::min<int64_t>(nb_, O_ - o0));
    const int gelems = nbk * gblk_;
    const int oelems = nbk * oblk_;
    const scalar_t* gsrc = grad_ + o0 * gblk_;
    scalar_t* odst = out_ + o0 * oblk_;

    sycl::uint4* s4 = slm_.template get_multi_ptr<sycl::access::decorated::no>().get();
    scalar_t* s = reinterpret_cast<scalar_t*>(s4);

    // Phase 1: contiguous grad chunk -> SLM, all loads of a work-item before its SLM stores.
    {
      const int g4 = gelems / VO;
      const sycl::uint4* gsrc4 = reinterpret_cast<const sycl::uint4*>(gsrc);
      sycl::uint4 v[LPL];
#pragma unroll
      for (int k = 0; k < LPL; ++k) {
        const int idx = lid + k * kBlockWG;
        if (idx < g4) v[k] = gsrc4[idx];
      }
#pragma unroll
      for (int k = 0; k < LPL; ++k) {
        const int idx = lid + k * kBlockWG;
        if (idx < g4) s4[idx] = v[k];
      }
      const int t = g4 * VO + lid;  // element tail of a partial last chunk (< VO elements)
      if (lid < VO && t < gelems) s[t] = gsrc[t];
    }
    sycl::group_barrier(it.get_group());

    // Phase 2: each work-item produces VO consecutive grad_input elements per unit.
    const int nunits = (oelems + VO - 1) / VO;
    sycl::uint4* odst4 = reinterpret_cast<sycl::uint4*>(odst);
    for (int u = lid; u < nunits; u += kBlockWG) {
      const uint32_t e0 = static_cast<uint32_t>(u) * VO;
      uint32_t o = div_dj_.div(e0);
      const uint32_t r = e0 - o * static_cast<uint32_t>(oblk_);
      uint32_t i = div_j_.div(r);
      uint32_t j = r - i * static_cast<uint32_t>(J_);
      int base = static_cast<int>(o) * gblk_ + static_cast<int>(j) * size_ + static_cast<int>(i);
      scalar_t res[VO];
#pragma unroll
      for (int v = 0; v < VO; ++v) {
        int fhi = sycl::min(static_cast<int>(div_step_.div(i)), F_ - 1);
        const int t = static_cast<int>(i) + step_ - size_;
        const int flo = t >= 0 ? static_cast<int>(div_step_.div(static_cast<uint32_t>(t))) : 0;
        if (static_cast<int>(e0) + v >= oelems) fhi = -1;
        scalar_t acc = scalar_t(0);
        int idx = base + flo * fstride_;
        for (int f = flo; f <= fhi; ++f) {
          acc += s[idx];
          idx += fstride_;
        }
        res[v] = acc;
        ++j;
        base += size_;
        if (j == static_cast<uint32_t>(J_)) {
          j = 0;
          ++i;
          if (i == static_cast<uint32_t>(D_)) {
            i = 0;
            ++o;
          }
          base = static_cast<int>(o) * gblk_ + static_cast<int>(i);
        }
      }
      if (static_cast<int>(e0) + VO <= oelems) {
        sycl::uint4 w;
        std::memcpy(&w, res, sizeof(w));
        odst4[u] = w;
      } else {
#pragma unroll
        for (int v = 0; v < VO; ++v)
          if (static_cast<int>(e0) + v < oelems) odst[e0 + v] = res[v];
      }
    }
  }

  const scalar_t* grad_;
  scalar_t* out_;
  int64_t O_;
  int nb_;
  int gblk_;    // F*J*size
  int oblk_;    // D*J
  int J_, D_, F_, size_, step_;
  int fstride_; // J*size - step
  Div32 div_dj_, div_j_, div_step_;
  sycl::local_accessor<sycl::uint4, 1> slm_;
};

template <typename scalar_t>
struct UnfoldBwdGenericKernel {
  void operator()(sycl::nd_item<1> it) const {
    const int64_t e = static_cast<int64_t>(it.get_global_linear_id());
    if (e >= N_) return;
    const int64_t o = e / DJ_;
    const int64_t r = e - o * DJ_;
    const int64_t i = r / J_;
    const int64_t j = r - i * J_;
    const int64_t fhi = sycl::min<int64_t>(i / step_, F_ - 1);
    const int64_t t = i + step_ - size_;
    const int64_t flo = t >= 0 ? t / step_ : 0;
    scalar_t acc = scalar_t(0);
    for (int64_t f = flo; f <= fhi; ++f)
      acc += grad_[o * gblk_ + f * (J_ * size_ - step_) + j * size_ + i];
    out_[e] = acc;
  }
  const scalar_t* grad_;
  scalar_t* out_;
  int64_t N_, DJ_, J_, F_, size_, step_, gblk_;
};

void unfold_backward_kernel(
    Tensor& grad_out,
    const Tensor& grad_in,
    int64_t dim,
    int64_t size,
    int64_t step) {
  dim = maybe_wrap_dim(dim, grad_out.dim());

  TORCH_CHECK(step > 0 && step < size, "unfold_backward: requires 0 < step < size");
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  TORCH_CHECK(grad_in.scalar_type() == grad_out.scalar_type(), "dtype mismatch");

  const int64_t nd = static_cast<int64_t>(grad_out.dim());
  int64_t O = 1, J = 1;
  for (int64_t d = 0; d < dim; ++d) O *= grad_out.size(d);
  for (int64_t d = dim + 1; d < nd; ++d) J *= grad_out.size(d);
  const int64_t D = grad_out.size(dim);
  const int64_t F = (D - size) / step + 1;
  const int64_t N = O * D * J;
  if (N == 0) return;

  Tensor g = grad_in.is_contiguous() ? grad_in : grad_in.contiguous();
  auto& q = getCurrentSYCLQueue();
  const int64_t es = g.element_size();
  const int64_t gblk = F * J * size;
  const int64_t gbytes = gblk * es;
  const int64_t obytes = D * J * es;
  const int64_t na = std::max<int64_t>(64 / std::gcd<int64_t>(64, obytes), 16 / std::gcd<int64_t>(16, gbytes));
  const int64_t nb = (kSlmBytes / gbytes) / na * na;
  const bool aligned = (reinterpret_cast<uintptr_t>(g.data_ptr()) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(grad_out.data_ptr()) % 16 == 0);
  const int64_t i32max = std::numeric_limits<int32_t>::max();
  const bool block_path = nb >= 1 && aligned && g.numel() <= i32max && N <= i32max;

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, g.scalar_type(), "unfold_backward_xpu", [&] {
    const scalar_t* gp = g.const_data_ptr<scalar_t>();
    scalar_t* op = grad_out.data_ptr<scalar_t>();
    if (block_path) {
      using K = UnfoldBwdBlockKernel<scalar_t>;
      const int64_t nchunks = (O + nb - 1) / nb;
      q.submit([&](sycl::handler& cgh) {
        K k;
        k.grad_ = gp;
        k.out_ = op;
        k.O_ = O;
        k.nb_ = static_cast<int>(nb);
        k.gblk_ = static_cast<int>(gblk);
        k.oblk_ = static_cast<int>(D * J);
        k.J_ = static_cast<int>(J);
        k.D_ = static_cast<int>(D);
        k.F_ = static_cast<int>(F);
        k.size_ = static_cast<int>(size);
        k.step_ = static_cast<int>(step);
        k.fstride_ = static_cast<int>(J * size - step);
        k.div_dj_ = Div32(static_cast<uint32_t>(D * J));
        k.div_j_ = Div32(static_cast<uint32_t>(J));
        k.div_step_ = Div32(static_cast<uint32_t>(step));
        k.slm_ = sycl::local_accessor<sycl::uint4, 1>(sycl::range<1>(kSlmU4), cgh);
        cgh.parallel_for<K>(
            sycl::nd_range<1>(sycl::range<1>(nchunks * kBlockWG), sycl::range<1>(kBlockWG)), k);
      });
    } else {
      using K = UnfoldBwdGenericKernel<scalar_t>;
      K k{gp, op, N, D * J, J, F, size, step, gblk};
      const int64_t wg = 256;
      const int64_t groups = (N + wg - 1) / wg;
      sycl_kernel_submit(groups * wg, wg, q, k);
    }
  });
}
} // namespace at::native::xpu
