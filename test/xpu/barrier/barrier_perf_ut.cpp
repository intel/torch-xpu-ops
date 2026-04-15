#include "barrier_impl_xpu.hpp"

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<sycl::device> get_level_zero_gpus() {
  std::vector<sycl::device> devices;
  for (const auto& platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    for (const auto& device : platform.get_devices()) {
      if (device.is_gpu()) {
        devices.push_back(device);
      }
    }
  }
  if (devices.empty()) {
    throw std::runtime_error("No Level Zero GPU device found");
  }
  return devices;
}

void print_l0_device_info(const sycl::queue& q, int rank) {
  ze_device_handle_t ze_dev =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());

  ze_device_properties_t props;
  std::memset(&props, 0, sizeof(props));
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ze_result_t ret = zeDeviceGetProperties(ze_dev, &props);
  if (ret != ZE_RESULT_SUCCESS) {
    std::cout << "[barrier_ut] rank=" << rank
              << " zeDeviceGetProperties failed, ze_result=" << ret << std::endl;
    return;
  }

  std::cout << "[barrier_ut] rank=" << rank << " L0 device: " << props.name
            << ", timerResolution(ns): " << props.timerResolution
            << ", vendorId: " << props.vendorId << std::endl;
}

struct SignalPads {
  sycl::context ctx;
  std::vector<uint32_t*> device_pad_ptrs;
  std::vector<uint32_t**> device_pad_tables;
  std::vector<sycl::device> devices;

  SignalPads(
      const sycl::context& context,
      const std::vector<sycl::device>& devs,
      const std::vector<sycl::queue>& queues,
      int channels)
      : ctx(context), devices(devs) {
    const int world_size = static_cast<int>(devices.size());
    device_pad_ptrs.resize(world_size, nullptr);
    device_pad_tables.resize(world_size, nullptr);

    const size_t entries_per_rank = static_cast<size_t>(world_size) * channels;
    for (int rank = 0; rank < world_size; ++rank) {
      device_pad_ptrs[rank] =
          sycl::malloc_device<uint32_t>(entries_per_rank, devices[rank], ctx);
      if (device_pad_ptrs[rank] == nullptr) {
        throw std::runtime_error("malloc_device failed for signal pad");
      }
      queues[rank].memset(
          device_pad_ptrs[rank], 0, entries_per_rank * sizeof(uint32_t));
    }

    for (int rank = 0; rank < world_size; ++rank) {
      device_pad_tables[rank] =
          sycl::malloc_device<uint32_t*>(world_size, devices[rank], ctx);
      if (device_pad_tables[rank] == nullptr) {
        throw std::runtime_error("malloc_device failed for signal pad ptr table");
      }
      queues[rank].memcpy(
          device_pad_tables[rank],
          device_pad_ptrs.data(),
          static_cast<size_t>(world_size) * sizeof(uint32_t*));
    }
    for (const auto& q : queues) {
      q.wait_and_throw();
    }
  }

  ~SignalPads() {
    for (uint32_t** table : device_pad_tables) {
      if (table != nullptr) {
        sycl::free(table, ctx);
      }
    }
    for (uint32_t* ptr : device_pad_ptrs) {
      if (ptr != nullptr) {
        sycl::free(ptr, ctx);
      }
    }
  }
};

double benchmark_barrier_us(
    std::vector<sycl::queue>& queues,
    const SignalPads& pads,
    int world_size,
    int channel,
    int warmup,
    int iters) {
  auto launch_one_iteration = [&]() {
    std::vector<sycl::event> events;
    events.reserve(static_cast<size_t>(world_size));
    for (int rank = 0; rank < world_size; ++rank) {
      events.push_back(test_xpu_barrier::barrier_impl_xpu(
          pads.device_pad_tables[rank],
          channel,
          rank,
          world_size,
          10'000,
          queues[rank]));
    }
    for (auto& event : events) {
      event.wait_and_throw();
    }
  };

  for (int i = 0; i < warmup; ++i) {
    launch_one_iteration();
  }

  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    launch_one_iteration();
  }
  const auto end = std::chrono::steady_clock::now();

  const auto total_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  return static_cast<double>(total_us) / static_cast<double>(iters);
}

int get_env_int(const char* name, int default_value) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return default_value;
  }
  return std::atoi(v);
}

} // namespace

int main() {
  try {
    auto all_devices = get_level_zero_gpus();

    const int world_size =
        get_env_int("XPU_BARRIER_WORLD_SIZE", static_cast<int>(all_devices.size()));
    const int channels = 1;
    const int channel = 0;
    const int warmup = get_env_int("XPU_BARRIER_PERF_WARMUP", 100);
    const int iters = get_env_int("XPU_BARRIER_PERF_ITERS", 1000);

    if (world_size <= 1) {
      throw std::runtime_error("XPU_BARRIER_WORLD_SIZE must be > 1");
    }
    if (world_size > static_cast<int>(all_devices.size())) {
      throw std::runtime_error(
          "XPU_BARRIER_WORLD_SIZE exceeds number of available L0 GPU devices");
    }
    if (warmup < 0 || iters <= 0) {
      throw std::runtime_error("Invalid warmup/iters settings");
    }

    std::vector<sycl::device> devices(
        all_devices.begin(), all_devices.begin() + world_size);
    sycl::context ctx(devices);

    std::vector<sycl::queue> queues;
    queues.reserve(static_cast<size_t>(world_size));
    for (int rank = 0; rank < world_size; ++rank) {
      queues.emplace_back(ctx, devices[rank], sycl::property::queue::in_order());
      print_l0_device_info(queues.back(), rank);
    }

    SignalPads pads(ctx, devices, queues, channels);

    const double avg_us = benchmark_barrier_us(
        queues, pads, world_size, channel, warmup, iters);

    std::cout << "[barrier_ut] impl=barrier_impl_xpu_multi_device"
              << ", world_size=" << world_size << ", warmup=" << warmup
              << ", iters=" << iters << ", avg_latency_us=" << avg_us
              << std::endl;

    std::cout << "[barrier_ut] PASS" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[barrier_ut] FAIL: " << e.what() << std::endl;
    return 1;
  }
}
