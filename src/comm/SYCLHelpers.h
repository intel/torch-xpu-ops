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
#include <comm/Macros.h>
DISABLE_SYCL_DEPRECATED_WARNING_BEGIN
// Official suppression macro provided by Intel SYCL headers for
// host-only compilation (without -fsycl).
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include <comm/Scalar.h>
#include <sycl/sycl.hpp>
#undef SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
DISABLE_SYCL_DEPRECATED_WARNING_END

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

// sycl access address space
static constexpr auto sycl_priv_space =
    sycl::access::address_space::private_space;
static constexpr auto sycl_local_space =
    sycl::access::address_space::local_space;
static constexpr auto sycl_global_space =
    sycl::access::address_space::global_space;

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
  requires std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>
static inline void sycl_kernel_submit(
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
  requires(!std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>)
static inline void sycl_kernel_submit(
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

template <typename ker_t>
  requires std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>
static inline void sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        ker);
  };
  q.submit(cgf);
}

template <typename ker_t>
  requires(!std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>)
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

// Overloads accepting kernel properties (e.g., sub_group_size, grf_size).
// Uses kernel functor with get(properties_tag) — the official non-deprecated
// way to attach compile-time properties to a kernel.

template <typename KernelType, typename PropsType>
struct __SyclKernelWithProps__ {
  KernelType kernel_;
  template <typename ItemT>
  void operator()(ItemT&& item) const {
    kernel_(std::forward<ItemT>(item));
  }
  template <typename ItemT, typename... Rest>
  void operator()(ItemT&& item, Rest&&...) const {
    kernel_(std::forward<ItemT>(item));
  }
  auto get(::sycl::ext::oneapi::experimental::properties_tag) const {
    return PropsType{};
  }
};

template <typename ker_t, typename Props, int dim>
  requires std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>
static inline void sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    Props properties,
    ker_t ker) {
  (void)properties;
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    __SyclKernelWithProps__<ker_t, Props> wrapped{ker};
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<dim>(global_range, local_range), wrapped);
  };
  q.submit(cgf);
}

template <typename ker_t, typename Props, int dim>
  requires(!std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>)
static inline void sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    Props properties,
    ker_t ker) {
  (void)properties;
  auto cgf = [&](::sycl::handler& cgh) {
    __SyclKernelWithProps__<ker_t, Props> wrapped{ker};
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<dim>(global_range, local_range), wrapped);
  };
  q.submit(cgf);
}

template <typename ker_t, typename Props>
  requires std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>
static inline void sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    Props properties,
    ker_t ker) {
  (void)properties;
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    __SyclKernelWithProps__<ker_t, Props> wrapped{ker};
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        wrapped);
  };
  q.submit(cgf);
}

template <typename ker_t, typename Props>
  requires(!std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>)
static inline void sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    Props properties,
    ker_t ker) {
  (void)properties;
  auto cgf = [&](::sycl::handler& cgh) {
    __SyclKernelWithProps__<ker_t, Props> wrapped{ker};
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        wrapped);
  };
  q.submit(cgf);
}

// For SYCL free function
// Submit a SYCL free function kernel via sycl_ext_oneapi_free_function_kernels.
// See:
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_free_function_kernels.asciidoc
//
// The spec defines a struct template and a corresponding variable template:
//   template<auto *Func> struct kernel_function_s {};
//   template<auto *Func> inline constexpr kernel_function_s<Func>
//   kernel_function;
//
// `kernel_function<kptr>` is an inline constexpr instance of
// `kernel_function_s<kptr>`. We use it instead of explicitly constructing
// `kernel_function_s<kptr>{}` because:
//   1. It is the idiomatic shorthand encouraged by the spec examples
//      (e.g. `syclexp::nd_launch(q, ndr, syclexp::kernel_function<iota>,
//      ...)`).
//   2. Both forms are equivalent — `kernel_function<kptr>` simply evaluates to
//      a default-constructed `kernel_function_s<kptr>` at compile time.

// TODO: unify and remove the if-else for slm_sz
template <auto* kptr, typename... Kargs>
static inline void sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    int slm_sz,
    Kargs... args) {
#if defined(SYCL_COMPILER_VERSION) && SYCL_COMPILER_VERSION < 20260100
  sycl::context ctxt = q.get_context();
  auto exe_bndl =
      syclexp::get_kernel_bundle<kptr, sycl::bundle_state::executable>(ctxt);
  sycl::kernel ker = exe_bndl.template ext_oneapi_get_kernel<kptr>();
  if (slm_sz != 0) {
    syclexp::launch_config cfg{
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        syclexp::properties{syclexp::work_group_scratch_size(slm_sz)}};
    syclexp::nd_launch(q, cfg, ker, args...);
  } else {
    syclexp::launch_config cfg{::sycl::nd_range<1>(
        ::sycl::range<1>(global_range), ::sycl::range<1>(local_range))};
    syclexp::nd_launch(q, cfg, ker, args...);
  }
#else
  if (slm_sz != 0) {
    syclexp::launch_config cfg{
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        syclexp::properties{syclexp::work_group_scratch_size(slm_sz)}};
    syclexp::nd_launch(q, cfg, syclexp::kernel_function<kptr>, args...);
  } else {
    syclexp::nd_launch(
        q,
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        syclexp::kernel_function<kptr>,
        args...);
  }
#endif
}

// TODO: unify and remove the if-else for slm_sz
template <auto* kptr, int dim, typename... Kargs>
static inline void sycl_kernel_submit(
    ::sycl::range<dim> global_range,
    ::sycl::range<dim> local_range,
    ::sycl::queue q,
    int slm_sz,
    Kargs... args) {
#if defined(SYCL_COMPILER_VERSION) && SYCL_COMPILER_VERSION < 20260100
  sycl::context ctxt = q.get_context();
  auto exe_bndl =
      syclexp::get_kernel_bundle<kptr, sycl::bundle_state::executable>(ctxt);
  sycl::kernel ker = exe_bndl.template ext_oneapi_get_kernel<kptr>();
  if (slm_sz != 0) {
    syclexp::launch_config cfg{
        ::sycl::nd_range<dim>(
            ::sycl::range<dim>(global_range), ::sycl::range<dim>(local_range)),
        syclexp::properties{syclexp::work_group_scratch_size(slm_sz)}};
    syclexp::nd_launch(q, cfg, ker, args...);
  } else {
    syclexp::launch_config cfg{::sycl::nd_range<dim>(
        ::sycl::range<dim>(global_range), ::sycl::range<dim>(local_range))};
    syclexp::nd_launch(q, cfg, ker, args...);
  }
#else
  if (slm_sz != 0) {
    syclexp::launch_config cfg{
        ::sycl::nd_range<dim>(global_range, local_range),
        syclexp::properties{syclexp::work_group_scratch_size(slm_sz)}};
    syclexp::nd_launch(q, cfg, syclexp::kernel_function<kptr>, args...);
  } else {
    syclexp::nd_launch(
        q,
        ::sycl::nd_range<dim>(global_range, local_range),
        syclexp::kernel_function<kptr>,
        args...);
  }
#endif
}

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_KERNEL_STRING(var, str) \
  static const __attribute__((opencl_constant)) char var[] = str
#else
#define SYCL_KERNEL_STRING(var, str) static const char var[] = str
#endif
#define SYCL_KERNEL_PRINTF sycl::ext::oneapi::experimental::printf

#define SYCL_PRINT(fmt_str, ...)                \
  {                                             \
    SYCL_KERNEL_STRING(fmt_var, fmt_str);       \
    SYCL_KERNEL_PRINTF(fmt_var, ##__VA_ARGS__); \
  }

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_REQD_SUB_GROUP_SIZE(x) [[sycl::reqd_sub_group_size(x)]]
#else
#define SYCL_REQD_SUB_GROUP_SIZE(x)
#endif
