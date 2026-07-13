/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/NumericUtils.h>
#include <comm/SYCLHelpers.h>

#include <sycl/sycl.hpp>

#include <concepts>
#include <type_traits>

namespace at::native::xpu {

template <typename T>
static inline T safe_max(T a, T b) {
  T max = at::_isnan(a) ? a : (at::_isnan(b) ? b : std::max<T>(a, b));
  return max;
}

template <typename T>
static inline T safe_min(T a, T b) {
  T min = at::_isnan(a) ? a : (at::_isnan(b) ? b : std::min<T>(a, b));
  return min;
}

template <typename T>
using sycl_atomic_ref_rlx_wg_local_t =
    sycl::atomic_ref<T, sycl_mem_odr_rlx, sycl_mem_scp_wg, sycl_local_space>;

// The integer atomics differ between the global and work-group-local address
// spaces only in the atomic_ref they build, so they are parameterized on it.
// Types narrower than a word are packed several-per-word and CAS the containing
// 32-bit word; wider ones CAS their own word directly.
template <template <typename> class atomic_ref_t, std::integral T>
struct AtomicIntegerImplBase {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    if constexpr (sizeof(T) == 1) {
      size_t offset = (size_t)address & 3;
      uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
      uint32_t assumed = *address_as_ui;
      uint32_t shift = offset * 8;
      uint32_t newval;
      atomic_ref_t<uint32_t> target(*address_as_ui);

      do {
        uint32_t old_byte = (assumed >> shift) & 0xff;
        // preserve size in initial cast. Casting directly to uint32_t pads
        // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
        newval = static_cast<uint8_t>(func(val, static_cast<T>(old_byte)));
        newval = (assumed & ~(0x000000ff << shift)) | (newval << shift);
      } while (!target.compare_exchange_strong(assumed, newval));
    } else if constexpr (sizeof(T) == 2) {
      size_t offset = (size_t)address & 2;
      uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
      bool is_32_align = offset;
      uint32_t assumed = *address_as_ui;
      uint32_t newval;
      atomic_ref_t<uint32_t> target(*address_as_ui);

      do {
        uint32_t old_half = is_32_align ? assumed >> 16 : assumed & 0xffff;
        // preserve size in initial cast. Casting directly to uint32_t pads
        // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
        newval = static_cast<uint16_t>(func(val, static_cast<T>(old_half)));
        newval = is_32_align ? (assumed & 0xffff) | (newval << 16)
                             : (assumed & 0xffff0000) | newval;
      } while (!target.compare_exchange_strong(assumed, newval));
    } else {
      using proxy_t =
          std::conditional_t<sizeof(T) == 4, uint32_t, unsigned long long>;
      proxy_t* address_as_proxy = (proxy_t*)address;
      proxy_t assumed = *address_as_proxy;
      proxy_t newval;
      atomic_ref_t<proxy_t> target(*address_as_proxy);

      do {
        newval = static_cast<proxy_t>(func(val, static_cast<T>(assumed)));
      } while (!target.compare_exchange_strong(assumed, newval));
    }
  }
};

template <typename T>
using AtomicIntegerImplLocal =
    AtomicIntegerImplBase<sycl_atomic_ref_rlx_wg_local_t, T>;

template <typename T>
using AtomicIntegerImpl =
    AtomicIntegerImplBase<sycl_atomic_ref_rlx_dev_global_t, T>;

#define SYCL_ATOMIC_INTEGER_LOCAL(NAME, OP, DTYPE)          \
  static inline void atomic##NAME##Local(                   \
      const sycl_local_ptr<DTYPE>& address, DTYPE val) {    \
    AtomicIntegerImplLocal<DTYPE>()(                        \
        address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

#define SYCL_ATOMIC_INTEGER(NAME, OP, DTYPE)                \
  static inline void atomic##NAME(                          \
      const sycl_global_ptr<DTYPE>& address, DTYPE val) {   \
    AtomicIntegerImpl<DTYPE>()(                             \
        address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

// For operations sycl::atomic_ref supports natively on 4/8-byte integers.
#define SYCL_ATOMIC_INTEGER_NATIVE_IMPL(                                       \
    NAME, METHOD, DTYPE, PTR_TYPE, ATOMIC_REF)                                 \
  static inline void atomic##NAME(const PTR_TYPE<DTYPE>& address, DTYPE val) { \
    ATOMIC_REF<DTYPE> target(*address);                                        \
    target.METHOD(val);                                                        \
  }

#define SYCL_ATOMIC_INTEGER_NATIVE(NAME, METHOD, DTYPE) \
  SYCL_ATOMIC_INTEGER_NATIVE_IMPL(                      \
      NAME, METHOD, DTYPE, sycl_global_ptr, sycl_atomic_ref_rlx_dev_global_t)

#define SYCL_ATOMIC_INTEGER_NATIVE_LOCAL(NAME, METHOD, DTYPE) \
  SYCL_ATOMIC_INTEGER_NATIVE_IMPL(                            \
      NAME, METHOD, DTYPE, sycl_local_ptr, sycl_atomic_ref_rlx_wg_local_t)

template <typename T>
concept atomic_fp_t =
    std::same_as<T, at::Half> || std::same_as<T, at::BFloat16> ||
    std::same_as<T, float> || std::same_as<T, double>;

// CAS compares object representations, so NaN cannot livelock the loop.
// Half/BFloat16 are packed two-per-word and CAS the containing 32-bit word;
// float/double CAS their own word through a same-sized integer proxy.
template <template <typename> class atomic_ref_t, atomic_fp_t T>
struct AtomicFPImplBase {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    if constexpr (sizeof(T) == 2) {
      size_t offset = (size_t)address & 2;
      unsigned int* address_as_ui = (unsigned int*)((char*)address - offset);
      bool is_32_align = offset;
      unsigned int assumed = *address_as_ui;
      unsigned int newval;
      atomic_ref_t<unsigned int> target(*address_as_ui);

      do {
        uint16_t prev = static_cast<uint16_t>(
            is_32_align ? assumed >> 16 : assumed & 0xffff);
        uint16_t res =
            sycl::bit_cast<uint16_t>(func(sycl::bit_cast<T>(prev), val));
        newval = is_32_align
            ? (assumed & 0xffff) | (static_cast<unsigned int>(res) << 16)
            : (assumed & 0xffff0000) | res;
      } while (!target.compare_exchange_strong(assumed, newval));
    } else {
      using proxy_t =
          std::conditional_t<sizeof(T) == 4, unsigned int, unsigned long long>;
      proxy_t* address_as_proxy = (proxy_t*)address;
      proxy_t assumed = *address_as_proxy;
      proxy_t newval;
      atomic_ref_t<proxy_t> target(*address_as_proxy);

      do {
        newval = sycl::bit_cast<proxy_t>(func(val, sycl::bit_cast<T>(assumed)));
      } while (!target.compare_exchange_strong(assumed, newval));
    }
  }
};

template <typename T>
using AtomicFPImpl = AtomicFPImplBase<sycl_atomic_ref_rlx_dev_global_t, T>;

template <typename T>
using AtomicFPImplLocal = AtomicFPImplBase<sycl_atomic_ref_rlx_wg_local_t, T>;

#define SYCL_ATOMIC_FP(NAME, OP, DTYPE)                                       \
  static inline void atomic##NAME(                                            \
      const sycl_global_ptr<DTYPE>& address, DTYPE val) {                     \
    AtomicFPImpl<DTYPE>()(address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

#define SYCL_ATOMIC_FP_LOCAL(NAME, OP, DTYPE)               \
  static inline void atomic##NAME##Local(                   \
      const sycl_local_ptr<DTYPE>& address, DTYPE val) {    \
    AtomicFPImplLocal<DTYPE>()(                             \
        address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

static inline void atomicAdd(const sycl_global_ptr<float>& address, float val) {
  sycl_atomic_ref_rlx_dev_global_t<float> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_global_ptr<double>& address,
    double val) {
  sycl_atomic_ref_rlx_dev_global_t<double> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(const sycl_global_ptr<int>& address, int val) {
  sycl_atomic_ref_rlx_dev_global_t<int> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_global_ptr<int64_t>& address,
    int64_t val) {
  sycl_atomic_ref_rlx_dev_global_t<int64_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_local_ptr<uint32_t>& address,
    uint32_t val) {
  sycl_atomic_ref_rlx_wg_local_t<uint32_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_local_ptr<uint64_t>& address,
    uint64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<uint64_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(const sycl_local_ptr<int>& address, int val) {
  sycl_atomic_ref_rlx_wg_local_t<int> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_local_ptr<int64_t>& address,
    int64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<int64_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<float>& address,
    float val) {
  sycl_atomic_ref_rlx_wg_local_t<float> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<double>& address,
    double val) {
  sycl_atomic_ref_rlx_wg_local_t<double> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(const sycl_local_ptr<int>& address, int val) {
  sycl_atomic_ref_rlx_wg_local_t<int> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<int64_t>& address,
    int64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<int64_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<uint32_t>& address,
    uint32_t val) {
  sycl_atomic_ref_rlx_wg_local_t<uint32_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<uint64_t>& address,
    uint64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<uint64_t> target(*address);
  target.fetch_add(val);
}

// Atomic add local implementation.
SYCL_ATOMIC_INTEGER_LOCAL(Add, a || b, bool)
SYCL_ATOMIC_INTEGER_LOCAL(Add, std::plus<uint8_t>()(a, b), uint8_t)
SYCL_ATOMIC_INTEGER_LOCAL(Add, std::plus<int8_t>()(a, b), int8_t)
SYCL_ATOMIC_INTEGER_LOCAL(Add, std::plus<int16_t>()(a, b), int16_t)

SYCL_ATOMIC_FP_LOCAL(Add, std::plus<at::Half>()(a, b), at::Half)
SYCL_ATOMIC_FP_LOCAL(Add, std::plus<at::BFloat16>()(a, b), at::BFloat16)

// Atomic add implementation.
SYCL_ATOMIC_INTEGER(Add, a || b, bool)
SYCL_ATOMIC_INTEGER(Add, std::plus<uint8_t>()(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Add, std::plus<int8_t>()(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Add, std::plus<int16_t>()(a, b), int16_t)

SYCL_ATOMIC_FP(Add, std::plus<at::Half>()(a, b), at::Half)
SYCL_ATOMIC_FP(Add, std::plus<at::BFloat16>()(a, b), at::BFloat16)

template <typename T>
static inline void atomicAdd(
    const sycl_global_ptr<c10::complex<T>>& address,
    c10::complex<T> val) {
  atomicAdd(&address->real_, val.real_);
  atomicAdd(&address->imag_, val.imag_);
}

template <typename T>
static inline void atomicAddLocal(
    const sycl_local_ptr<c10::complex<T>>& address,
    c10::complex<T> val) {
  atomicAddLocal(&address->real_, val.real_);
  atomicAddLocal(&address->imag_, val.imag_);
}

// Atomic multiplication implementation.
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<uint8_t>()(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<int8_t>()(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<int16_t>()(a, b), int16_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<int32_t>()(a, b), int32_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<int64_t>()(a, b), int64_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<uint32_t>()(a, b), uint32_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<uint64_t>()(a, b), uint64_t)

SYCL_ATOMIC_FP(Mul, std::multiplies<float>()(a, b), float)
SYCL_ATOMIC_FP(Mul, std::multiplies<double>()(a, b), double)
SYCL_ATOMIC_FP(Mul, std::multiplies<at::Half>()(a, b), at::Half)
SYCL_ATOMIC_FP(Mul, std::multiplies<at::BFloat16>()(a, b), at::BFloat16)

// Atomic maximum implementation.

SYCL_ATOMIC_INTEGER_NATIVE_LOCAL(Max, fetch_max, int32_t)
SYCL_ATOMIC_INTEGER_NATIVE_LOCAL(Max, fetch_max, int64_t)

SYCL_ATOMIC_INTEGER(Max, safe_max<uint8_t>(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int8_t>(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int16_t>(a, b), int16_t)
SYCL_ATOMIC_INTEGER_NATIVE(Max, fetch_max, int32_t)
SYCL_ATOMIC_INTEGER_NATIVE(Max, fetch_max, int64_t)
SYCL_ATOMIC_INTEGER_NATIVE(Max, fetch_max, uint32_t)
SYCL_ATOMIC_INTEGER_NATIVE(Max, fetch_max, uint64_t)

SYCL_ATOMIC_FP(Max, safe_max<float>(a, b), float)
SYCL_ATOMIC_FP(Max, safe_max<double>(a, b), double)
SYCL_ATOMIC_FP(Max, safe_max<at::Half>(a, b), at::Half)
SYCL_ATOMIC_FP(Max, safe_max<at::BFloat16>(a, b), at::BFloat16)

// Atomic minimum implementation.
SYCL_ATOMIC_INTEGER(Min, safe_min<uint8_t>(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int8_t>(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int16_t>(a, b), int16_t)
SYCL_ATOMIC_INTEGER_NATIVE(Min, fetch_min, int32_t)
SYCL_ATOMIC_INTEGER_NATIVE(Min, fetch_min, int64_t)
SYCL_ATOMIC_INTEGER_NATIVE(Min, fetch_min, uint32_t)
SYCL_ATOMIC_INTEGER_NATIVE(Min, fetch_min, uint64_t)

SYCL_ATOMIC_FP(Min, safe_min<float>(a, b), float)
SYCL_ATOMIC_FP(Min, safe_min<double>(a, b), double)
SYCL_ATOMIC_FP(Min, safe_min<at::Half>(a, b), at::Half)
SYCL_ATOMIC_FP(Min, safe_min<at::BFloat16>(a, b), at::BFloat16)

// =========================================================================
// ------------------------------AtomicCAS----------------------------------
// =========================================================================

// atomicCAS is only used on 4/8-byte integer index types. sycl::atomic_ref
// supports them directly, and compare_exchange_strong compares object
// representations, matching CUDA atomicCAS bit semantics; expected is updated
// with the old value on failure, so it is the return value either way.
template <typename T, template <typename> class R>
struct AtomicCASImpl {
  inline T operator()(T* address, T expected, T desired) {
    R<T> target(*address);
    target.compare_exchange_strong(expected, desired);
    return expected;
  }
};

#define SYCL_ATOMIC_CAS_IMPL(DTYPE, PTR_TYPE, ATOMIC_REF)                  \
  static inline DTYPE atomicCAS(                                           \
      const PTR_TYPE<DTYPE>& address, DTYPE expected, DTYPE desired) {     \
    return AtomicCASImpl<DTYPE, ATOMIC_REF>()(address, expected, desired); \
  }

#define SYCL_ATOMIC_CAS_ALL(DTYPE) \
  /* local CAS version */          \
  SYCL_ATOMIC_CAS_IMPL(DTYPE, sycl_local_ptr, sycl_atomic_ref_rlx_wg_local_t)

SYCL_ATOMIC_CAS_ALL(int)
SYCL_ATOMIC_CAS_ALL(int64_t)
SYCL_ATOMIC_CAS_ALL(uint32_t)
SYCL_ATOMIC_CAS_ALL(uint64_t)

} // namespace at::native::xpu
