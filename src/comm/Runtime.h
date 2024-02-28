#pragma once

#include <c10/xpu/XPUStream.h>

namespace xpu {
namespace sycl {

static inline at::DeviceIndex syclGetDeviceIndexOfCurrentQueue() {
  return c10::xpu::getCurrentXPUStream().device_index();
}

static inline ::sycl::queue& syclGetCurrentQueue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

}} // namespace xpu::sycl
