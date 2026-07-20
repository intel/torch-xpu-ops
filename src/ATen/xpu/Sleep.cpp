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
#if SYCL_COMPILER_VERSION >= 20260000
  // SYCL free-function kernel compiled at runtime via the online-compilation
  // extension (sycl_ext_oneapi_kernel_compiler).
  static const std::string source = R"""(
    #include <sycl/sycl.hpp>

    namespace syclex = sycl::ext::oneapi::experimental;

    extern "C"
    SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::single_task_kernel))
    void spin_kernel(uint64_t cycles) {
      uint64_t start = syclex::clock<syclex::clock_scope::device>();
      while (syclex::clock<syclex::clock_scope::device>() - start < cycles) {}
    }
  )""";

  namespace syclex = sycl::ext::oneapi::experimental;
  // TODO: PVC does not expose sycl_ext_oneapi_device_clock because its bundled
  // IGC (Intel Graphics Compiler) is too old, and enabling the extension breaks
  // AOT builds on PVC. Online compilation is used here as a workaround. Once
  // the driver ships a newer IGC, migrate to a functor-based implementation.
  TORCH_CHECK_NOT_IMPLEMENTED(
      c10::xpu::get_raw_device(c10::xpu::current_device())
          .has(sycl::aspect::ext_oneapi_clock_device),
      "Requires the sycl_ext_oneapi_device_clock extension, "
      "which is not supported on this device. ",
      "Please upgrade to a newer driver.");
  auto& queue = getCurrentSYCLQueue();

  // All XPU queues in a process share the same SYCL context, so compile once
  // and reuse the kernel object across calls.
  static sycl::kernel spin_kernel = [&]() {
    auto kb_src = syclex::create_kernel_bundle_from_source(
        c10::xpu::get_device_context(), syclex::source_language::sycl, source);
    auto kb_exe = syclex::build(kb_src);
    return kb_exe.ext_oneapi_get_kernel("spin_kernel");
  }();

  queue.submit([&](sycl::handler& cgh) {
    cgh.set_args(cycles);
    cgh.single_task(spin_kernel);
  });
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "sleep is not supported for the current SYCL compiler version. ",
      "Please upgrade to SYCL compiler version 2026.0 or newer.");
#endif
}

} // namespace at::xpu
