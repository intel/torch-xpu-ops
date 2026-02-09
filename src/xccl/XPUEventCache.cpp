/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Suppress deprecation warnings from oneAPI SYCL headers.
// These are not from our code and would otherwise fail the build under -Werror.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include <c10/xpu/XPUStream.h>
#include <xccl/XPUEventCache.hpp>
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#pragma GCC diagnostic pop
#include <map>

namespace c10d {

XPUEventCache::XPUEventCache() = default;

std::shared_ptr<at::xpu::XPUEvent> XPUEventCache::create(bool timing) {
  auto deleter = [cache = shared_from_this(),
                  timing](at::xpu::XPUEvent* event) {
    std::lock_guard<std::mutex> lock(cache->cacheMutex_);

    cache->eventsArray_[timing ? 1 : 0].push_back(event);
  };
  at::xpu::XPUEvent* event = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    auto& events = eventsArray_[timing ? 1 : 0];
    // If we still have events in the cache, we reuse it. Otherwise, we create a
    // new one.
    if (!events.empty()) {
      event = events.front();
      events.pop_front();
    } else {
      event = new at::xpu::XPUEvent(timing);
    }
  }
  return std::shared_ptr<at::xpu::XPUEvent>(event, std::move(deleter));
}

std::shared_ptr<XPUEventCache> XPUEventCache::get(at::DeviceIndex device) {
  static thread_local std::map<at::DeviceIndex, std::shared_ptr<XPUEventCache>>
      cacheDeviceMap;
  // Check if device has already been in the map, if not, add a new entry
  auto it = cacheDeviceMap.find(device);
  if (it == cacheDeviceMap.end()) {
    cacheDeviceMap.emplace(device, std::make_shared<XPUEventCache>());
  }
  return cacheDeviceMap[device];
}

} // namespace c10d
