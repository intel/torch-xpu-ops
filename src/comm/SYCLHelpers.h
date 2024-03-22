#pragma once

#include <ATen/detail/FunctionTraits.h>

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
  // XXX: c10::xpu::getStreamFromPool().queue();
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
  // XXX: c10::xpu::getStreamFromPool().queue();
  q.submit(cgf);
}

template <typename ker_t>
static inline void sycl_kernel_submit(
    ::sycl::range<2> global_range,
    ::sycl::range<2> local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<2>(global_range, local_range), ker);
  };
  // XXX: c10::xpu::getStreamFromPool().queue();
  q.submit(cgf);
}

template <typename ker_t, typename ker_creator_t>
static inline void sycl_kernel_submit(
    ::sycl::range<2> global_range,
    ::sycl::range<2> local_range,
    ::sycl::queue q,
    ker_creator_t creator) {
  using traits = function_traits<ker_creator_t>;
  static_assert(
      std::is_same<ker_t, typename traits::result_type>::value,
      "Kernel type does not match with the return type of kernel creator ...");
  auto cgf = [&](::sycl::handler& cgh) {
    ker_t ker = creator(cgh);
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<2>(global_range, local_range), ker);
  };
  // XXX: c10::xpu::getStreamFromPool().queue();
  q.submit(cgf);
}
