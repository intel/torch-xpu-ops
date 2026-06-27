#pragma once

#include <c10/core/GeneratorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <utility>

// This shared library (xpu_hal.dll) stores function pointers that allow
// kernel DLLs built in BUILD_SEPARATE_OPS mode to call generator
// functions without linking torch_xpu.dll.  torch_xpu.dll registers
// the actual implementations at init time.
//
// Dependency graph:
//   torch_xpu.dll ──▶ xpu_hal.dll (registers at init)
//   kernel DLLs   ──▶ xpu_hal.dll (calls at runtime)
//
// No cycle: xpu_hal.dll only links c10, never torch_xpu.

namespace xpu_hal {

using GetDefaultGeneratorFn =
    c10::intrusive_ptr<c10::GeneratorImpl> (*)(int64_t device_index);
using PhiloxStateFn = std::pair<uint64_t, uint64_t> (*)(
    c10::GeneratorImpl* gen,
    uint64_t increment);

// Called once by torch_xpu during XPU generator initialization.
void registerXPUGeneratorBridge(
    GetDefaultGeneratorFn get_gen,
    PhiloxStateFn philox);

// Bridge accessors for kernel DLLs.
c10::intrusive_ptr<c10::GeneratorImpl> getDefaultGenerator(
    int64_t device_index);

std::pair<uint64_t, uint64_t> philoxState(
    c10::GeneratorImpl* gen,
    uint64_t increment);

} // namespace xpu_hal
