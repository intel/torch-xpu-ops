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
#include <c10/util/bit_cast.h>
#include <comm/SYCLHelpers.h>
#include <sycl/sycl.hpp>

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
using sycl_atomic_ref_rlx_dev_global_t =
    sycl::atomic_ref<T, sycl_mem_odr_rlx, sycl_mem_scp_dev, sycl_global_space>;

template <typename T>
using sycl_atomic_ref_rlx_wg_local_t =
    sycl::atomic_ref<T, sycl_mem_odr_rlx, sycl_mem_scp_wg, sycl_local_space>;

template <typename T, size_t n>
struct AtomicIntegerImplLocal;

template <typename T>
struct AtomicIntegerImplLocal<T, 1> {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    size_t offset = (size_t)address & 3;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    uint32_t assumed = *address_as_ui;
    uint32_t shift = offset * 8;
    uint32_t newval;
    uint32_t newval_byte;
    sycl_atomic_ref_rlx_wg_local_t<uint32_t> target(*address_as_ui);

    do {
      newval = assumed;
      newval_byte = (newval >> shift) & 0xff;
      // preserve size in initial cast. Casting directly to uint32_t pads
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint8_t>(func(val, static_cast<T>(newval_byte)));
      newval = (assumed & ~(0x000000ff << shift)) | (newval << shift);
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicIntegerImplLocal<T, 2> {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    size_t offset = (size_t)address & 2;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    bool is_32_align = offset;
    uint32_t assumed = *address_as_ui;
    uint32_t newval;
    uint32_t newval_bytes;
    sycl_atomic_ref_rlx_wg_local_t<uint32_t> target(*address_as_ui);

    do {
      newval = assumed;
      newval_bytes = is_32_align ? newval >> 16 : newval & 0xffff;
      // preserve size in initial cast. Casting directly to uint32_t pads
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint16_t>(func(val, static_cast<T>(newval_bytes)));
      newval = is_32_align ? (assumed & 0xffff) | (newval << 16)
                           : (assumed & 0xffff0000) | newval;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicIntegerImplLocal<T, 4> {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    uint32_t* address_as_ui = (uint32_t*)(address);
    uint32_t assumed = *address_as_ui;
    uint32_t newval;
    sycl_atomic_ref_rlx_wg_local_t<uint32_t> target(*address_as_ui);

    do {
      newval = static_cast<uint32_t>(func(val, static_cast<T>(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicIntegerImplLocal<T, 8> {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    unsigned long long* address_as_ull = (unsigned long long*)(address);
    unsigned long long assumed = *address_as_ull;
    unsigned long long newval;
    sycl_atomic_ref_rlx_wg_local_t<unsigned long long> target(*address_as_ull);

    do {
      newval = static_cast<uint64_t>(func(val, static_cast<T>(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

#define SYCL_ATOMIC_INTEGER_LOCAL(NAME, OP, DTYPE)          \
  static inline void atomic##NAME##Local(                   \
      const sycl_local_ptr<DTYPE>& address, DTYPE val) {    \
    AtomicIntegerImplLocal<DTYPE, sizeof(DTYPE)>()(         \
        address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

template <typename T, size_t n>
struct AtomicIntegerImpl;

template <typename T>
struct AtomicIntegerImpl<T, 1> {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    size_t offset = (size_t)address & 3;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    uint32_t assumed = *address_as_ui;
    uint32_t shift = offset * 8;
    uint32_t newval;
    uint32_t newval_byte;
    sycl_atomic_ref_rlx_dev_global_t<uint32_t> target(*address_as_ui);

    do {
      newval = assumed;
      newval_byte = (newval >> shift) & 0xff;
      // preserve size in initial cast. Casting directly to uint32_t pads
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint8_t>(func(val, static_cast<T>(newval_byte)));
      newval = (assumed & ~(0x000000ff << shift)) | (newval << shift);
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicIntegerImpl<T, 2> {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    size_t offset = (size_t)address & 2;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    bool is_32_align = offset;
    uint32_t assumed = *address_as_ui;
    uint32_t newval;
    uint32_t newval_bytes;
    sycl_atomic_ref_rlx_dev_global_t<uint32_t> target(*address_as_ui);

    do {
      newval = assumed;
      newval_bytes = is_32_align ? newval >> 16 : newval & 0xffff;
      // preserve size in initial cast. Casting directly to uint32_t pads
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint16_t>(func(val, static_cast<T>(newval_bytes)));
      newval = is_32_align ? (assumed & 0xffff) | (newval << 16)
                           : (assumed & 0xffff0000) | newval;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicIntegerImpl<T, 4> {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    uint32_t* address_as_ui = (uint32_t*)(address);
    uint32_t assumed = *address_as_ui;
    uint32_t newval;
    sycl_atomic_ref_rlx_dev_global_t<uint32_t> target(*address_as_ui);

    do {
      newval = static_cast<uint32_t>(func(val, static_cast<T>(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T>
struct AtomicIntegerImpl<T, 8> {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    unsigned long long* address_as_ull = (unsigned long long*)(address);
    unsigned long long assumed = *address_as_ull;
    unsigned long long newval;
    sycl_atomic_ref_rlx_dev_global_t<unsigned long long> target(
        *address_as_ull);

    do {
      newval = static_cast<uint64_t>(func(val, static_cast<T>(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

#define SYCL_ATOMIC_INTEGER(NAME, OP, DTYPE)                \
  static inline void atomic##NAME(                          \
      const sycl_global_ptr<DTYPE>& address, DTYPE val) {   \
    AtomicIntegerImpl<DTYPE, sizeof(DTYPE)>()(              \
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

// float/double are supported by sycl::atomic_ref directly;
// compare_exchange_strong compares object representations, so NaN cannot
// livelock the loop. Half/BFloat16 use the containing-word emulation below.
template <typename T>
struct AtomicFPImpl {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    T assumed = *address;
    sycl_atomic_ref_rlx_dev_global_t<T> target(*address);
    while (!target.compare_exchange_strong(assumed, func(val, assumed))) {
    }
  }
};

template <>
struct AtomicFPImpl<at::Half> {
  template <typename func_t>
  inline void operator()(at::Half* address, at::Half val, const func_t& func) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int assumed = *address_as_ui;
    unsigned int newval;
    sycl_atomic_ref_rlx_dev_global_t<unsigned int> target(*address_as_ui);

    do {
      newval = assumed;
      at::Half hsum;
      hsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
      hsum = func(hsum, val);
      newval = (size_t)address & 2 ? (newval & 0xffff) | (hsum.x << 16)
                                   : (newval & 0xffff0000) | hsum.x;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <>
struct AtomicFPImpl<at::BFloat16> {
  template <typename func_t>
  inline void operator()(
      at::BFloat16* address,
      at::BFloat16 val,
      const func_t& func) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int assumed = *address_as_ui;
    unsigned int newval;
    sycl_atomic_ref_rlx_dev_global_t<unsigned int> target(*address_as_ui);

    do {
      newval = assumed;
      at::BFloat16 bsum;
      bsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
      bsum = func(bsum, val);
      newval = (size_t)address & 2 ? (newval & 0xffff) | (bsum.x << 16)
                                   : (newval & 0xffff0000) | bsum.x;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

#define SYCL_ATOMIC_FP(NAME, OP, DTYPE)                                       \
  static inline void atomic##NAME(                                            \
      const sycl_global_ptr<DTYPE>& address, DTYPE val) {                     \
    AtomicFPImpl<DTYPE>()(address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

template <typename T>
struct AtomicFPImplLocal;

template <>
struct AtomicFPImplLocal<at::Half> {
  template <typename func_t>
  inline void operator()(at::Half* address, at::Half val, const func_t& func) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int assumed = *address_as_ui;
    unsigned int newval;
    sycl_atomic_ref_rlx_wg_local_t<unsigned int> target(*address_as_ui);

    do {
      newval = assumed;
      at::Half hsum;
      hsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
      hsum = func(hsum, val);
      newval = (size_t)address & 2 ? (newval & 0xffff) | (hsum.x << 16)
                                   : (newval & 0xffff0000) | hsum.x;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <>
struct AtomicFPImplLocal<at::BFloat16> {
  template <typename func_t>
  inline void operator()(
      at::BFloat16* address,
      at::BFloat16 val,
      const func_t& func) {
    unsigned int* address_as_ui =
        (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int assumed = *address_as_ui;
    unsigned int newval;
    sycl_atomic_ref_rlx_wg_local_t<unsigned int> target(*address_as_ui);

    do {
      newval = assumed;
      at::BFloat16 bsum;
      bsum.x = (size_t)address & 2 ? (newval >> 16) : (newval & 0xffff);
      bsum = func(bsum, val);
      newval = (size_t)address & 2 ? (newval & 0xffff) | (bsum.x << 16)
                                   : (newval & 0xffff0000) | bsum.x;
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

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

// --- Auxiliary Type Definition ---
// R is a template template parameter for the SYCL atomic ref type
template <typename T, template <typename> class R>
using AtomicRef = R<T>;

// --- Generic CAS Structure Definition (R is the Atomic Ref type) ---
// 4/8-byte types are supported by sycl::atomic_ref directly, and
// compare_exchange_strong compares object representations, matching CUDA
// atomicCAS bit semantics for NaN and +/-0; expected is updated with the
// old value on failure, so it is the return value either way. 1/2-byte
// types use the containing-word emulation below.
template <typename T, size_t n, template <typename> class R>
struct AtomicCASImpl {
  inline T operator()(T* address, T expected, T desired) {
    AtomicRef<T, R> target(*address);
    target.compare_exchange_strong(expected, desired);
    return expected;
  }
};

// n=1 (1-byte Soft-RMW)
template <typename T, template <typename> class R>
struct AtomicCASImpl<T, 1, R> {
  inline T operator()(T* address, T expected, T desired) {
    size_t offset = (size_t)address & 3;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    size_t shift = offset * 8;
    uint32_t assumed;
    uint32_t newval;
    AtomicRef<uint32_t, R> target(*address_as_ui);

    T extracted_old_value;
    do {
      assumed = *address_as_ui;
      uint32_t byte_in_mem = (assumed >> shift) & 0xff;
      extracted_old_value = static_cast<T>(byte_in_mem);

      if (extracted_old_value == expected) {
        uint32_t desired_byte = static_cast<uint8_t>(desired);
        newval = (assumed & ~(0x000000ff << shift)) | (desired_byte << shift);
      } else {
        break;
      }
    } while (!target.compare_exchange_strong(assumed, newval));

    if (extracted_old_value == expected) {
      return expected;
    } else {
      return extracted_old_value;
    }
  }
};

// n=2 (2-byte Soft-RMW, compared by object representation)
template <typename T, template <typename> class R>
struct AtomicCASImpl<T, 2, R> {
  inline T operator()(T* address, T expected, T desired) {
    size_t offset = (size_t)address & 2;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    bool is_upper_half = offset;
    uint32_t assumed;
    uint32_t newval;

    AtomicRef<uint32_t, R> target(*address_as_ui);

    const uint32_t expected_half_word = c10::bit_cast<uint16_t>(expected);
    const uint32_t desired_half_word = c10::bit_cast<uint16_t>(desired);
    uint32_t current_half_word;

    do {
      assumed = *address_as_ui;
      current_half_word = is_upper_half ? (assumed >> 16) : (assumed & 0xffff);

      if (current_half_word == expected_half_word) {
        newval = is_upper_half ? (assumed & 0xffff) | (desired_half_word << 16)
                               : (assumed & 0xffff0000) | desired_half_word;
      } else {
        break;
      }
    } while (!target.compare_exchange_strong(assumed, newval));

    if (current_half_word == expected_half_word) {
      return expected;
    }
    return c10::bit_cast<T>(static_cast<uint16_t>(current_half_word));
  }
};

// --- Generic Macro Definitions for Function Signatures ---
#define SYCL_ATOMIC_CAS_IMPL(DTYPE, PTR_TYPE, ATOMIC_REF)              \
  static inline DTYPE atomicCAS(                                       \
      const PTR_TYPE<DTYPE>& address, DTYPE expected, DTYPE desired) { \
    return AtomicCASImpl<DTYPE, sizeof(DTYPE), ATOMIC_REF>()(          \
        address, expected, desired);                                   \
  }

#define SYCL_ATOMIC_CAS_ALL(DTYPE) \
  /* local CAS version */          \
  SYCL_ATOMIC_CAS_IMPL(DTYPE, sycl_local_ptr, sycl_atomic_ref_rlx_wg_local_t)

SYCL_ATOMIC_CAS_ALL(int)
SYCL_ATOMIC_CAS_ALL(int64_t)
SYCL_ATOMIC_CAS_ALL(uint32_t)
SYCL_ATOMIC_CAS_ALL(uint64_t)
SYCL_ATOMIC_CAS_ALL(int8_t)
SYCL_ATOMIC_CAS_ALL(uint8_t)
SYCL_ATOMIC_CAS_ALL(float)
SYCL_ATOMIC_CAS_ALL(double)
SYCL_ATOMIC_CAS_ALL(at::Half)
SYCL_ATOMIC_CAS_ALL(at::BFloat16)

} // namespace at::native::xpu
