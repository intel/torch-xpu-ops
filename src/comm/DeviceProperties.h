#pragma once

#include <ATen/xpu/XPUContext.h>

#include <comm/Runtime.h>
#include <sycl/sycl.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace xpu {
namespace sycl {

template <class KernelClass>
static int64_t syclMaxWorkGroupSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto& ctx = c10::xpu::get_device_context();
  auto& dev = c10::xpu::get_raw_device(dev_id);

  auto kid = ::sycl::get_kernel_id<KernelClass>();
  // The kernel won't be built for devices except for the first device.
  // Launching kernel on devices except for the first device will raise
  // runtime error. Here is an alternative as a temporary solution to
  // provide an extra hint to SYCL runtime.
  // https://github.com/intel/llvm/issues/15127
  auto kbundle = ::sycl::get_kernel_bundle<::sycl::bundle_state::executable>(
      ctx, {dev}, {kid});

  ::sycl::kernel k = kbundle.get_kernel(kid);
  return k.get_info<::sycl::info::kernel_device_specific::work_group_size>(dev);
}

template <class KernelClass>
static int64_t syclMaxWorkGroupSize(
    const KernelClass& /*kfn*/,
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  return syclMaxWorkGroupSize<KernelClass>(dev_id);
}

// For SYCL free function
template <auto* kptr>
static int64_t syclMaxWorkGroupSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto q = c10::xpu::getCurrentXPUStream(dev_id).queue();
  auto ctxt = q.get_context();
  auto dev = q.get_device();
  auto exe_bndl =
      ::syclexp::get_kernel_bundle<kptr, ::sycl::bundle_state::executable>(
          ctxt);
  ::sycl::kernel k = exe_bndl.template ext_oneapi_get_kernel<kptr>();
  return k.get_info<::sycl::info::kernel_device_specific::work_group_size>(dev);
}

static inline int64_t syclDeviceMaxWorkGroupSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->max_work_group_size;
}

static inline int64_t syclMaxSubGroupSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  const auto& subgroup_sizes = dev_prop->sub_group_sizes;
  TORCH_CHECK(
      !subgroup_sizes.empty(),
      "The device subgroup sizes is empty, please check the device status.");
  return *std::max_element(subgroup_sizes.begin(), subgroup_sizes.end());
}

static inline int64_t syclMinSubGroupSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  const auto& subgroup_sizes = dev_prop->sub_group_sizes;
  TORCH_CHECK(
      !subgroup_sizes.empty(),
      "The device subgroup sizes is empty, please check the device status.");
  return *std::min_element(subgroup_sizes.begin(), subgroup_sizes.end());
}

static inline int64_t syclMaxComputeUnitSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->max_compute_units;
}

static inline int64_t syclGpuEuCount(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_eu_count;
}

static inline int64_t syclGpuEuSimdWidth(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_eu_simd_width;
}

static inline int64_t syclGpuHWThreadsPerEU(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_hw_threads_per_eu;
}

static inline int64_t syclGpuEUCountPerSubslice(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->gpu_eu_count_per_subslice;
}

static inline int64_t syclMaxWorkItemsPerTile(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t eu_cnt = dev_prop->gpu_eu_count;
  int64_t simd_width = syclMaxSubGroupSize(dev_id);
  int64_t hw_threads = dev_prop->gpu_hw_threads_per_eu;
  return eu_cnt * simd_width * hw_threads;
}

static inline int64_t syclMaxWorkItemsPerSubSlice(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t simd_width = syclMaxSubGroupSize(dev_id);
  int64_t eu_count = dev_prop->gpu_eu_count_per_subslice;
  return simd_width * eu_count;
}

static inline int64_t syclMaxWorkItemsPerEU(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  int64_t simd_width = syclMaxSubGroupSize(dev_id);
  int64_t hw_threads = dev_prop->gpu_hw_threads_per_eu;
  return simd_width * hw_threads;
}

static inline int64_t syclMaxNumSubGroups(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->max_num_sub_groups;
}

static inline int64_t syclMaxDSSNum(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  int64_t dss_num =
      syclMaxComputeUnitSize(dev_id) / syclGpuEUCountPerSubslice(dev_id);
  return dss_num;
}

static inline size_t syclGlobalMemSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->global_mem_size;
}

static inline int64_t syclLocalMemSize(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->local_mem_size;
}

template <typename T>
uint32_t syclPrefVectorWidth(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  (void)dev_id; // Suppress unused variable warning

  // Hot fix. This is the preferred vector width for GPUs up to LNL/BMG.
  constexpr uint32_t vec_width = 16;

  if constexpr (
      std::is_same_v<T, char> || std::is_same_v<T, short> ||
      std::is_same_v<T, int> || std::is_same_v<T, int64_t> ||
      std::is_same_v<T, float> || std::is_same_v<T, double> ||
      std::is_same_v<T, ::sycl::half>) {
    return vec_width / sizeof(T);
  } else {
    throw std::invalid_argument(
        "Invalid data type to fetch preferred vector width!");
  }
}

template <typename T>
uint32_t syclNativeVectorWidth(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  if constexpr (std::is_same_v<T, char>) {
    return dev_prop->native_vector_width_char;
  } else if constexpr (std::is_same_v<T, short>) {
    return dev_prop->native_vector_width_short;
  } else if constexpr (std::is_same_v<T, int>) {
    return dev_prop->native_vector_width_int;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return dev_prop->native_vector_width_long;
  } else if constexpr (std::is_same_v<T, float>) {
    return dev_prop->native_vector_width_float;
  } else if constexpr (std::is_same_v<T, double>) {
    return dev_prop->native_vector_width_double;
  } else if constexpr (std::is_same_v<T, ::sycl::half>) {
    return dev_prop->native_vector_width_half;
  } else {
    throw std::invalid_argument(
        "Invalid data type to fetch native vector width!");
  }
}

static inline bool syclHasFloat64(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  auto* dev_prop = at::xpu::getDeviceProperties(dev_id);
  return dev_prop->has_fp64;
}

} // namespace sycl
} // namespace xpu
