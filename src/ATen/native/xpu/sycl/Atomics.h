#pragma once

#include <ATen/NumericUtils.h>
#include <comm/SYCLHelpers.h>
#include <comm/Scalar.h>
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
  static inline void atomic##NAME(                          \
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

template <typename T>
struct AtomicFPImpl;

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

template <>
struct AtomicFPImpl<float> {
  template <typename func_t>
  inline void operator()(float* address, float val, const func_t& func) {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int assumed = *address_as_ui;
    unsigned int newval;
    sycl_atomic_ref_rlx_dev_global_t<unsigned int> target(*address_as_ui);

    do {
      newval = __float_as_int(func(val, __int_as_float(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <>
struct AtomicFPImpl<double> {
  template <typename func_t>
  inline void operator()(double* address, double val, const func_t& func) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long assumed = *address_as_ull;
    unsigned long long newval;
    sycl_atomic_ref_rlx_dev_global_t<unsigned long long> target(
        *address_as_ull);

    do {
      newval = __double_as_long_long(func(val, __long_long_as_double(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

#define SYCL_ATOMIC_FP(NAME, OP, DTYPE)                                       \
  static inline void atomic##NAME(                                            \
      const sycl_global_ptr<DTYPE>& address, DTYPE val) {                     \
    AtomicFPImpl<DTYPE>()(address, val, [](DTYPE a, DTYPE b) { return OP; }); \
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

static inline void atomicMax(
    const sycl_local_ptr<int32_t>& address,
    int32_t val) {
  sycl_atomic_ref_rlx_wg_local_t<int32_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicMax(
    const sycl_local_ptr<int64_t>& address,
    int64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<int64_t> target(*address);
  target.fetch_add(val);
}

SYCL_ATOMIC_INTEGER_LOCAL(Max, safe_max<uint8_t>(a, b), uint8_t)
SYCL_ATOMIC_INTEGER_LOCAL(Max, safe_max<int8_t>(a, b), int8_t)
SYCL_ATOMIC_INTEGER_LOCAL(Max, safe_max<int16_t>(a, b), int16_t)

SYCL_ATOMIC_INTEGER(Max, safe_max<uint8_t>(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int8_t>(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int16_t>(a, b), int16_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int32_t>(a, b), int32_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int64_t>(a, b), int64_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<uint32_t>(a, b), uint32_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<uint64_t>(a, b), uint64_t)

SYCL_ATOMIC_FP(Max, safe_max<float>(a, b), float)
SYCL_ATOMIC_FP(Max, safe_max<double>(a, b), double)
SYCL_ATOMIC_FP(Max, safe_max<at::Half>(a, b), at::Half)
SYCL_ATOMIC_FP(Max, safe_max<at::BFloat16>(a, b), at::BFloat16)

// Atomic minimum implementation.
SYCL_ATOMIC_INTEGER(Min, safe_min<uint8_t>(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int8_t>(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int16_t>(a, b), int16_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int32_t>(a, b), int32_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int64_t>(a, b), int64_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<uint32_t>(a, b), uint32_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<uint64_t>(a, b), uint64_t)

SYCL_ATOMIC_FP(Min, safe_min<float>(a, b), float)
SYCL_ATOMIC_FP(Min, safe_min<double>(a, b), double)
SYCL_ATOMIC_FP(Min, safe_min<at::Half>(a, b), at::Half)
SYCL_ATOMIC_FP(Min, safe_min<at::BFloat16>(a, b), at::BFloat16)

} // namespace at::native::xpu
