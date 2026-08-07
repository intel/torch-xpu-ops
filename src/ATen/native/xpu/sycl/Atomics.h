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
#include <functional>
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

// CAS-emulated atomics: types narrower than a word are packed several-per-word
// and CAS the containing 32-bit word; wider ones CAS their own word directly.
template <template <typename> class atomic_ref_t, std::integral T>
struct AtomicIntegerImplBase {
  // Pack an n-byte value (n in {1,2}) into its containing 32-bit word and CAS.
  template <int n, typename func_t>
  static inline void subWord(T* address, T val, const func_t& func) {
    using sub_t = std::conditional_t<n == 1, uint8_t, uint16_t>;
    constexpr uint32_t mask = (1u << (n * 8)) - 1;
    size_t offset = (size_t)address & (4 - n);
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    uint32_t shift = offset * 8;
    uint32_t assumed = *address_as_ui;
    uint32_t newval = 0;
    atomic_ref_t<uint32_t> target(*address_as_ui);

    do {
      uint32_t old = (assumed >> shift) & mask;
      // Cast through sub_t, not uint32_t, so negative signed values do not
      // sign-extend (e.g. signed -1 stays 0xff, not ~0).
      newval = static_cast<sub_t>(func(val, static_cast<T>(old)));
      newval = (assumed & ~(mask << shift)) | (newval << shift);
    } while (!target.compare_exchange_strong(assumed, newval));
  }

  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    if constexpr (sizeof(T) == 1) {
      subWord<1>(address, val, func);
    } else if constexpr (sizeof(T) == 2) {
      subWord<2>(address, val, func);
    } else {
      using proxy_t =
          std::conditional_t<sizeof(T) == 4, uint32_t, unsigned long long>;
      proxy_t* address_as_proxy = (proxy_t*)address;
      proxy_t assumed = *address_as_proxy;
      proxy_t newval = 0;
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

template <typename T>
concept atomic_fp_t =
    std::same_as<T, at::Half> || std::same_as<T, at::BFloat16> ||
    std::same_as<T, float> || std::same_as<T, double>;

// CAS compares object representations, so NaN cannot livelock the loop.
template <template <typename> class atomic_ref_t, atomic_fp_t T>
struct AtomicFPImplBase {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    if constexpr (sizeof(T) == 2) {
      size_t offset = (size_t)address & 2;
      unsigned int* address_as_ui = (unsigned int*)((char*)address - offset);
      bool is_32_align = offset;
      unsigned int assumed = *address_as_ui;
      unsigned int newval = 0;
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
      proxy_t newval = 0;
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

// 4/8-byte integers: native sycl::atomic_ref fetch_*.
template <typename T>
concept atomic_native_int_t =
    std::integral<T> && (sizeof(T) == 4 || sizeof(T) == 8);

// compare_exchange_strong compares object representations; expected is updated
// with the old value on failure, so it is returned either way.
template <typename T, template <typename> class R>
struct AtomicCASImpl {
  inline T operator()(T* address, T expected, T desired) {
    R<T> target(*address);
    target.compare_exchange_strong(expected, desired);
    return expected;
  }
};

// Free-function atomics, overloaded on address space (sycl_global_ptr vs
// sycl_local_ptr). Native fetch_* where supported, else the CAS-emulated cores.

// Atomic add.
template <typename T>
  requires(std::integral<T> || atomic_fp_t<T>)
static inline void atomicAdd(const sycl_global_ptr<T>& address, T val) {
  if constexpr (
      atomic_native_int_t<T> || std::is_same_v<T, float> ||
      std::is_same_v<T, double>) {
    sycl_atomic_ref_rlx_dev_global_t<T> target(*address);
    target.fetch_add(val);
  } else if constexpr (std::integral<T>) {
    AtomicIntegerImpl<T>()(address, val, std::plus<T>());
  } else {
    AtomicFPImpl<T>()(address, val, std::plus<T>());
  }
}

template <typename T>
  requires(std::integral<T> || atomic_fp_t<T>)
static inline void atomicAdd(const sycl_local_ptr<T>& address, T val) {
  if constexpr (
      atomic_native_int_t<T> || std::is_same_v<T, float> ||
      std::is_same_v<T, double>) {
    sycl_atomic_ref_rlx_wg_local_t<T> target(*address);
    target.fetch_add(val);
  } else if constexpr (std::integral<T>) {
    AtomicIntegerImplLocal<T>()(address, val, std::plus<T>());
  } else {
    AtomicFPImplLocal<T>()(address, val, std::plus<T>());
  }
}

template <typename T>
static inline void atomicAdd(
    const sycl_global_ptr<c10::complex<T>>& address,
    c10::complex<T> val) {
  atomicAdd(sycl_global_ptr<T>(&address->real_), val.real_);
  atomicAdd(sycl_global_ptr<T>(&address->imag_), val.imag_);
}

template <typename T>
static inline void atomicAdd(
    const sycl_local_ptr<c10::complex<T>>& address,
    c10::complex<T> val) {
  atomicAdd(sycl_local_ptr<T>(&address->real_), val.real_);
  atomicAdd(sycl_local_ptr<T>(&address->imag_), val.imag_);
}

// Atomic multiply (no native fetch_mul; always CAS-emulated).
template <typename T>
  requires(std::integral<T> || atomic_fp_t<T>)
static inline void atomicMul(const sycl_global_ptr<T>& address, T val) {
  if constexpr (std::integral<T>) {
    AtomicIntegerImpl<T>()(address, val, std::multiplies<T>());
  } else {
    AtomicFPImpl<T>()(address, val, std::multiplies<T>());
  }
}

template <typename T>
  requires(std::integral<T> || atomic_fp_t<T>)
static inline void atomicMul(const sycl_local_ptr<T>& address, T val) {
  if constexpr (std::integral<T>) {
    AtomicIntegerImplLocal<T>()(address, val, std::multiplies<T>());
  } else {
    AtomicFPImplLocal<T>()(address, val, std::multiplies<T>());
  }
}

// Atomic maximum.
template <typename T>
  requires(std::integral<T> || atomic_fp_t<T>)
static inline void atomicMax(const sycl_global_ptr<T>& address, T val) {
  if constexpr (atomic_native_int_t<T>) {
    sycl_atomic_ref_rlx_dev_global_t<T> target(*address);
    target.fetch_max(val);
  } else if constexpr (std::integral<T>) {
    AtomicIntegerImpl<T>()(
        address, val, [](T a, T b) { return safe_max<T>(a, b); });
  } else {
    AtomicFPImpl<T>()(address, val, [](T a, T b) { return safe_max<T>(a, b); });
  }
}

template <typename T>
  requires(std::integral<T> || atomic_fp_t<T>)
static inline void atomicMax(const sycl_local_ptr<T>& address, T val) {
  if constexpr (atomic_native_int_t<T>) {
    sycl_atomic_ref_rlx_wg_local_t<T> target(*address);
    target.fetch_max(val);
  } else if constexpr (std::integral<T>) {
    AtomicIntegerImplLocal<T>()(
        address, val, [](T a, T b) { return safe_max<T>(a, b); });
  } else {
    AtomicFPImplLocal<T>()(
        address, val, [](T a, T b) { return safe_max<T>(a, b); });
  }
}

// Atomic minimum.
template <typename T>
  requires(std::integral<T> || atomic_fp_t<T>)
static inline void atomicMin(const sycl_global_ptr<T>& address, T val) {
  if constexpr (atomic_native_int_t<T>) {
    sycl_atomic_ref_rlx_dev_global_t<T> target(*address);
    target.fetch_min(val);
  } else if constexpr (std::integral<T>) {
    AtomicIntegerImpl<T>()(
        address, val, [](T a, T b) { return safe_min<T>(a, b); });
  } else {
    AtomicFPImpl<T>()(address, val, [](T a, T b) { return safe_min<T>(a, b); });
  }
}

template <typename T>
  requires(std::integral<T> || atomic_fp_t<T>)
static inline void atomicMin(const sycl_local_ptr<T>& address, T val) {
  if constexpr (atomic_native_int_t<T>) {
    sycl_atomic_ref_rlx_wg_local_t<T> target(*address);
    target.fetch_min(val);
  } else if constexpr (std::integral<T>) {
    AtomicIntegerImplLocal<T>()(
        address, val, [](T a, T b) { return safe_min<T>(a, b); });
  } else {
    AtomicFPImplLocal<T>()(
        address, val, [](T a, T b) { return safe_min<T>(a, b); });
  }
}

// Atomic compare-and-swap, work-group-local only.
template <typename T>
  requires std::integral<T> && (sizeof(T) == 4 || sizeof(T) == 8)
static inline T
    atomicCAS(const sycl_local_ptr<T>& address, T expected, T desired) {
  return AtomicCASImpl<T, sycl_atomic_ref_rlx_wg_local_t>()(
      address, expected, desired);
}

} // namespace at::native::xpu
