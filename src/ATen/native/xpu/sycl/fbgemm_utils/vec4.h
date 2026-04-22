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
 * BSD License
 * 
 * For FBGEMM software
 * 
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 *  * Neither the name Facebook nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

namespace fbgemm_utils {

////////////////////////////////////////////////////////////////////////////////
// Vec4T Base
////////////////////////////////////////////////////////////////////////////////

// Customized 4-element vector data types (with element type Half, or float).
template <typename T>
struct Vec4BaseT {
  sycl::float4 acc;

  Vec4BaseT() {
    acc= { 0.0, 0.0, 0.0, 0.0 };
  }

};

template <typename T>
struct Vec4T {};

// A wrapper for Vec4T with acc_type
template <typename T>
using Vec4TAcc = Vec4T<at::acc_type<T, true>>;

////////////////////////////////////////////////////////////////////////////////
// Vec4T<float>
////////////////////////////////////////////////////////////////////////////////

template <>
struct Vec4T<float> : public Vec4BaseT<float> {
  Vec4T() {}

  Vec4T(const float* p) {
    load(p);
  }

  Vec4T(const at::Half* p) {
    load(p);
  }

  Vec4T(const at::BFloat16* p) {
    load(p);
  }

  Vec4T(const c10::Float8_e4m3fnuz* p) {
    load(p);
  }

  void load(const float* p) {
    acc = *((const sycl::float4*)p);
  }

  void load(const at::Half* p) {
    // Convert half precision to float (SYCL version)
    sycl::half2 h0, h1;
    h0 = *reinterpret_cast<const sycl::half2*>(p);
    h1 = *reinterpret_cast<const sycl::half2*>(p + 2);
    
    acc.x() = static_cast<float>(h0.x());
    acc.y() = static_cast<float>(h0.y());
    acc.z() = static_cast<float>(h1.x());
    acc.w() = static_cast<float>(h1.y());
  }

  void load(const at::BFloat16* p) {
    acc.x() = static_cast<float>(p[0]);
    acc.y() = static_cast<float>(p[1]);
    acc.z() = static_cast<float>(p[2]);
    acc.w() = static_cast<float>(p[3]);
  }

  void load(const c10::Float8_e4m3fnuz* p) {
    assert(false && "Loading Float8_e4m3fnuz into Vec4T<float> is not supported.");
  }

  void load(const uint8_t* p) {
    assert(false && "Loading uint8_t into Vec4T<float> is not supported.");
  }

  void store(float* p) const {
    *((sycl::float4*)p) = acc;
  }

  void store(sycl::float4* p) const {
    *p = acc;
  }

  void store(at::Half* p) const {
    // Equivalent to CUDA __float22half2_rn(...): round-to-nearest-even
    sycl::vec<float, 2> a(acc.x(), acc.y());
    sycl::vec<float, 2> b(acc.z(), acc.w());

    auto ah = a.convert<sycl::half, sycl::rounding_mode::rte>();
    auto bh = b.convert<sycl::half, sycl::rounding_mode::rte>();

    p[0] = static_cast<at::Half>(ah[0]);
    p[1] = static_cast<at::Half>(ah[1]);
    p[2] = static_cast<at::Half>(bh[0]);
    p[3] = static_cast<at::Half>(bh[1]);
  }

  void store(at::BFloat16* p) const {
    p[0] = static_cast<at::BFloat16>(acc.x());
    p[1] = static_cast<at::BFloat16>(acc.y());
    p[2] = static_cast<at::BFloat16>(acc.z());
    p[3] = static_cast<at::BFloat16>(acc.w());
  }

  void store(uint8_t* p) const {
    assert(false && "Storing uint8_t from Vec4T<float> is not supported.");
  }

  void add_(const Vec4T<float>& a) {
    acc.x() += a.acc.x();
    acc.y() += a.acc.y();
    acc.z() += a.acc.z();
    acc.w() += a.acc.w();
  }

  void mul_(float scale) {
    acc.x() *= scale;
    acc.y() *= scale;
    acc.z() *= scale;
    acc.w() *= scale;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Vec4T<at::Half>
////////////////////////////////////////////////////////////////////////////////

template <>
struct Vec4T<at::Half> : public Vec4BaseT<at::Half> {
  Vec4T() {}

  Vec4T(const at::Half* p) {
    load(p);
  }
  Vec4T(const at::BFloat16* p) {
    load(p);
  }

  Vec4T(const c10::Float8_e4m3fnuz* p) {
    load(p);
  }

  Vec4T(const float* p) {
    load(p);
  }

  void load(const at::Half* p) {
    // Convert half precision to float
    sycl::half2 h0, h1;
    h0 = *reinterpret_cast<const sycl::half2*>(p);
    h1 = *reinterpret_cast<const sycl::half2*>(p + 2);
    
    acc.x() = static_cast<float>(h0.x());
    acc.y() = static_cast<float>(h0.y());
    acc.z() = static_cast<float>(h1.x());
    acc.w() = static_cast<float>(h1.y());
  }
  
  void load(const at::BFloat16* p) {
    acc.x() = static_cast<float>(p[0]);
    acc.y() = static_cast<float>(p[1]);
    acc.z() = static_cast<float>(p[2]);
    acc.w() = static_cast<float>(p[3]);
  }

 void load(const float* p) {
    acc = *((const sycl::float4*)p);
  }

  void load(const c10::Float8_e4m3fnuz* p) {
    assert(false && "Loading Float8_e4m3fnuz into Vec4T<at::Half> is not supported.");
  }

  void load(const uint8_t* p) {
    assert(false && "Loading uint8_t into Vec4T<at::Half> is not supported.");
  }

  void store(at::Half* p) const {
    // Equivalent to CUDA __float22half2_rn(...): round-to-nearest-even
    sycl::vec<float, 2> a(acc.x(), acc.y());
    sycl::vec<float, 2> b(acc.z(), acc.w());

    auto ah = a.convert<sycl::half, sycl::rounding_mode::rte>();
    auto bh = b.convert<sycl::half, sycl::rounding_mode::rte>();

    p[0] = static_cast<at::Half>(ah[0]);
    p[1] = static_cast<at::Half>(ah[1]);
    p[2] = static_cast<at::Half>(bh[0]);
    p[3] = static_cast<at::Half>(bh[1]);
  }

  void store(at::BFloat16* p) const {
    p[0] = static_cast<at::BFloat16>(acc.x());
    p[1] = static_cast<at::BFloat16>(acc.y());
    p[2] = static_cast<at::BFloat16>(acc.z());
    p[3] = static_cast<at::BFloat16>(acc.w());
  }

  void store(float* p) const {
    *((sycl::float4*)p) = acc;
  }

  void store(uint8_t* p) const {
    assert(false && "Storing uint8_t from Vec4T<at::Half> is not supported.");
  }

  void add_(const Vec4T<at::Half>& a) {
    acc.x() += a.acc.x();
    acc.y() += a.acc.y();
    acc.z() += a.acc.z();
    acc.w() += a.acc.w();
  }

  void mul_(float scale) {
    acc.x() *= scale;
    acc.y() *= scale;
    acc.z() *= scale;
    acc.w() *= scale;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Vec4T Ops
////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
Vec4T<scalar_t> vec4_acc(
    const Vec4T<scalar_t>& lhs,
    const Vec4T<scalar_t>& rhs) {
  Vec4T<scalar_t> s;
  s.acc.x() = lhs.acc.x() + rhs.acc.x();
  s.acc.y() = lhs.acc.y() + rhs.acc.y();
  s.acc.z() = lhs.acc.z() + rhs.acc.z();
  s.acc.w() = lhs.acc.w() + rhs.acc.w();
  return s;
}

} // namespace fbgemm_utils
