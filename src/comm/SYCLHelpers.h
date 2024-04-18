#pragma once

#include <ATen/detail/FunctionTraits.h>
#include <comm/Scalar.h>
#include <sycl/sycl.hpp>

// sycl access address space
static constexpr auto sycl_priv_space =
    sycl::access::address_space::private_space;
static constexpr auto sycl_local_space =
    sycl::access::address_space::local_space;
static constexpr auto sycl_global_space =
    sycl::access::address_space::global_space;

// sycl access fence space
static constexpr auto sycl_local_fence = sycl::access::fence_space::local_space;
static constexpr auto sycl_global_fence =
    sycl::access::fence_space::global_space;
static constexpr auto sycl_global_and_local_fence =
    sycl::access::fence_space::global_and_local;

// sycl memory ordering
static constexpr auto sycl_mem_odr_rlx = sycl::memory_order::relaxed;
static constexpr auto sycl_mem_odr_acq = sycl::memory_order::acquire;
static constexpr auto sycl_mem_odr_rel = sycl::memory_order::release;
static constexpr auto sycl_mem_odr_acq_rel = sycl::memory_order::acq_rel;
static constexpr auto sycl_mem_odr_seq_cst = sycl::memory_order::seq_cst;

// sycl memory scope
static constexpr auto sycl_mem_scp_wi = sycl::memory_scope::work_item;
static constexpr auto sycl_mem_scp_sg = sycl::memory_scope::sub_group;
static constexpr auto sycl_mem_scp_wg = sycl::memory_scope::work_group;
static constexpr auto sycl_mem_scp_dev = sycl::memory_scope::device;
static constexpr auto sycl_mem_scp_sys = sycl::memory_scope::system;

template <typename scalar_t, int dims = 1>
using sycl_local_acc_t = sycl::local_accessor<scalar_t, dims>;

template <typename T>
using sycl_local_ptr = typename sycl::local_ptr<T>;

template <typename T>
using sycl_global_ptr = typename sycl::global_ptr<T>;

template <typename T>
using sycl_atomic_ref_rlx_dev_global_t =
    sycl::atomic_ref<T, sycl_mem_odr_rlx, sycl_mem_scp_dev, sycl_global_space>;

template <typename ker_t>
static inline void sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        ker);
  };
  q.submit(cgf);
}

// Call for kernels using shared memory. The current SYCL command group handler
// is required to create shared memory (SYCL local accessor).
// To use sycl::ker_creator_t to define a creator for kernel.
template <typename ker_t, typename ker_creator_t>
static inline void sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    ker_creator_t creator) {
  using traits = function_traits<ker_creator_t>;
  static_assert(
      std::is_same<ker_t, typename traits::result_type>::value,
      "Kernel type does not match with the return type of kernel creator ...");
  auto cgf = [&](::sycl::handler& cgh) {
    ker_t ker = creator(cgh);
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        ker);
  };
  q.submit(cgf);
}

template <typename ker_t, int dim>
static inline void sycl_kernel_submit(
    ::sycl::range<dim> range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) { cgh.parallel_for<ker_t>(range, ker); };
  q.submit(cgf);
}

// Additional convention of SYCL kernel configuration. Besides construct kernel
// functor, SYCL has some additional conventions to be called during setuping
// SYCL command group handler, e.g. declaring SYCL local accessor when the
// kernel requires shared local memory usage. Helpers below help simpilfiy
// submission of SYCL kernels requiring additional conventions.

// Defining additional convention. Can use `sycl_kernel_submit` simply to
// submit a kernel, if the kernel functor inherits from the struct below.
// Since cannot offload non-device-copyable (sycl::is_device_copyable) kernel
// functor, a structure has virtual function is non-device-copyable.
// Using an empty class, the kernel functor derived by it will be required to
// define member method `void convention(sycl::handler&)`, or fails in
// compilation.
struct __SYCL_KER_CONFIG_CONVENTION__ {};

template <typename ker_t, int dim>
static inline typename std::enable_if<
    std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<dim>(global_range, local_range), ker);
  };
  q.submit(cgf);
}

template <typename ker_t, int dim>
static inline typename std::enable_if<
    !std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<dim>(global_range, local_range), ker);
  };
  q.submit(cgf);
}

// ================= atomic =================

template <typename T>
using sycl_atomic_ref_rlx_dev_global_t =
    sycl::atomic_ref<T, sycl_mem_odr_rlx, sycl_mem_scp_dev, sycl_global_space>;

template <typename T>
using sycl_atomic_ref_rlx_wg_local_t =
    sycl::atomic_ref<T, sycl_mem_odr_rlx, sycl_mem_scp_wg, sycl_local_space>;

template <typename T, typename func_t, int size>
struct AtomicIntegerImpl;

template <typename T, typename func_t>
struct AtomicIntegerImpl<T, func_t, 1> {
  void operator()(T* address, T val, func_t func) const {
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

template <typename T, typename func_t>
struct AtomicIntegerImpl<T, func_t, 2> {
  void operator()(T* address, T val, func_t func) const {
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

template <typename T, typename func_t>
struct AtomicIntegerImpl<T, func_t, 4> {
  void operator()(T* address, T val, func_t func) const {
    uint32_t* address_as_ui = (uint32_t*)(address);
    uint32_t assumed = *address_as_ui;
    uint32_t newval;
    sycl_atomic_ref_rlx_dev_global_t<uint32_t> target(*address_as_ui);

    do {
      newval = static_cast<uint32_t>(func(val, static_cast<T>(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename T, typename func_t>
struct AtomicIntegerImpl<T, func_t, 8> {
  void operator()(T* address, T val, func_t func) const {
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

#define SYCL_ATOMIC_INTEGER(NAME, OP, DTYPE)                             \
  static inline void atomic##NAME(                                       \
      const sycl_global_ptr<DTYPE>& address, DTYPE val) {                \
    auto caller = OP();                                                  \
    AtomicIntegerImpl<DTYPE, OP, sizeof(DTYPE)>()(address, val, caller); \
  }

template <typename T, typename func_t>
struct AtomicFPImpl;

template <typename func_t>
struct AtomicFPImpl<at::Half, func_t> {
  void operator()(at::Half* address, at::Half val, func_t func) const {
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

template <typename func_t>
struct AtomicFPImpl<at::BFloat16, func_t> {
  void operator()(at::BFloat16* address, at::BFloat16 val, func_t func) const {
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

template <typename func_t>
struct AtomicFPImpl<float, func_t> {
  void operator()(float* address, float val, func_t func) const {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int assumed = *address_as_ui;
    unsigned int newval;
    sycl_atomic_ref_rlx_dev_global_t<unsigned int> target(*address_as_ui);

    do {
      newval = __float_as_int(func(val, __int_as_float(assumed)));
    } while (!target.compare_exchange_strong(assumed, newval));
  }
};

template <typename func_t>
struct AtomicFPImpl<double, func_t> {
  void operator()(double* address, double val, func_t func) const {
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

#define SYCL_ATOMIC_FP(NAME, OP, DTYPE)                   \
  static inline void atomic##NAME(                        \
      const sycl_global_ptr<DTYPE>& address, DTYPE val) { \
    auto caller = OP();                                   \
    AtomicFPImpl<DTYPE, OP>()(address, val, caller);      \
  }

template <typename T>
struct NumericAdd {
  T operator()(T a, T b) const {
    return a + b;
  }
};

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

SYCL_ATOMIC_FP(Add, NumericAdd<at::Half>, at::Half)
SYCL_ATOMIC_FP(Add, NumericAdd<at::BFloat16>, at::BFloat16)

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

SYCL_ATOMIC_INTEGER(Add, NumericAdd<uint8_t>, uint8_t)
SYCL_ATOMIC_INTEGER(Add, NumericAdd<int8_t>, int8_t)
SYCL_ATOMIC_INTEGER(Add, NumericAdd<int16_t>, int16_t)

static inline void atomicAdd(const sycl_global_ptr<bool>& address, bool val) {
  *address = address && val;
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

template <typename T>
static inline void atomicAdd(
    const sycl_global_ptr<c10::complex<T>>& address,
    c10::complex<T> val) {
  atomicAdd(&address->real_, val.real_);
  atomicAdd(&address->imag_, val.imag_);
}
