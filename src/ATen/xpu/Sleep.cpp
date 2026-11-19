/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/xpu/Sleep.h>
#include <c10/xpu/XPUStream.h>
#include <comm/SYCLContext.h>

#if !defined(SYCL_COMPILER_VERSION)
#error \
    "SYCL_COMPILER_VERSION is not defined. Ensure SYCLToolkit is found before building torch-xpu-ops."
#endif

namespace at::xpu {

void sleep(uint64_t cycles) {
  // SYCL free-function kernels compiled at runtime via the online-compilation
  // extension (sycl_ext_oneapi_kernel_compiler).
  static const std::string clock_source = R"""(
    #include <sycl/sycl.hpp>

    namespace syclex = sycl::ext::oneapi::experimental;

    extern "C"
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::single_task_kernel))
    void spin_kernel(uint64_t cycles) {
      uint64_t start = syclex::clock<syclex::clock_scope::sub_group>();
      while (syclex::clock<syclex::clock_scope::sub_group>() - start < cycles) {}
    }
  )""";

  // Fallback for devices without ext_oneapi_clock_sub_group: spin for a number
  // of loop iterations instead of clock cycles. An iteration costs many cycles,
  // so the iteration count is scaled by a factor calibrated once at runtime
  // (see below).
  static const std::string loop_source = R"""(
    #include <sycl/sycl.hpp>

    namespace syclex = sycl::ext::oneapi::experimental;

    extern "C"
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::single_task_kernel))
    void spin_kernel(uint64_t cycles) {
      for (volatile uint64_t i = 0; i < cycles; i = i + 1) {}
    }
  )""";

  namespace syclex = sycl::ext::oneapi::experimental;

  struct SpinKernel {
    sycl::kernel kernel;
    // Device cycles per loop iteration; 0 when the kernel counts real cycles.
    double cycles_per_iter;
  };

  // TODO: PVC does not expose ext_oneapi_clock_sub_group because its bundled
  // IGC (Intel Graphics Compiler) is too old, and enabling the extension breaks
  // AOT builds on PVC. Online compilation is used here as a workaround. Once
  // the driver ships a newer IGC, migrate to a functor-based implementation.
  //
  // All XPU queues in a process share the same SYCL context, so compile once
  // and reuse the kernel object across calls.
  static const SpinKernel spin_kernel = [&]() -> SpinKernel {
    auto& device = c10::xpu::get_raw_device(c10::xpu::current_device());
    auto& context = c10::xpu::get_device_context();
    bool has_clock_sub_group =
        device.has(sycl::aspect::ext_oneapi_clock_sub_group);
    const auto& source = has_clock_sub_group ? clock_source : loop_source;
    auto kb_src = syclex::create_kernel_bundle_from_source(
        context, syclex::source_language::sycl, source);
    auto kernel = syclex::build(kb_src).ext_oneapi_get_kernel("spin_kernel");
    if (has_clock_sub_group) {
      return {kernel, 0.0};
    }

    // Time a fixed number of iterations on a private profiling queue, so the
    // calibration does not wait on work already queued on the current stream,
    // and stays legal while that stream is being captured into an XPUGraph
    // (host waits and profiling queries on a recording queue throw). The first
    // launch is a warm-up that lets the GPU ramp up its frequency; each launch
    // takes about 50ms.
    constexpr uint64_t kCalibrationIters = 1 << 17;
    sycl::queue queue(
        context,
        device,
        {sycl::property::queue::in_order(),
         sycl::property::queue::enable_profiling()});
    auto launch = [&] {
      return queue.submit([&](sycl::handler& cgh) {
        cgh.set_args(kCalibrationIters);
        cgh.single_task(kernel);
      });
    };
    launch(); // warm-up
    auto event = launch();
    // get_profiling_info blocks until the command completes.
    auto ns =
        event.get_profiling_info<sycl::info::event_profiling::command_end>() -
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    // max_clock_frequency is in MHz, i.e. 1e-3 cycles per ns.
    double cycles_per_ns =
        device.get_info<sycl::info::device::max_clock_frequency>() * 1e-3;
    double cycles_per_iter = ns * cycles_per_ns / kCalibrationIters;
    return {kernel, std::max(cycles_per_iter, 1.0)};
  }();

  uint64_t count = spin_kernel.cycles_per_iter > 0
      ? static_cast<uint64_t>(cycles / spin_kernel.cycles_per_iter)
      : cycles;
  getCurrentSYCLQueue().submit([&](sycl::handler& cgh) {
    cgh.set_args(count);
    cgh.single_task(spin_kernel.kernel);
  });
}

} // namespace at::xpu
