/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
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
