#pragma once

// Backward-compatible bridge header for generator functions.
//
// When <c10/xpu/XPUGeneratorBridge.h> is available (pytorch with the c10_xpu
// bridge PR), use the real bridge functions that route through c10_xpu.dll to
// avoid a link-time dependency on torch_xpu.dll.
//
// When the bridge is not available (building against an older pytorch), fall
// back to inline wrappers that call torch_xpu directly.  In BUILD_SEPARATE_OPS
// mode the linker needs /FORCE:UNRESOLVED for these symbols because kernel
// DLLs don't link torch_xpu.dll.

#if __has_include(<c10/xpu/XPUGeneratorBridge.h>)
#include <c10/xpu/XPUGeneratorBridge.h>
#else
#include <ATen/core/Generator.h>
#include <ATen/xpu/PhiloxXpuState.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/core/GeneratorImpl.h>
#include <utility>

namespace c10::xpu {

inline c10::intrusive_ptr<c10::GeneratorImpl>
getDefaultXPUGeneratorBridge(int64_t device_index) {
  return c10::intrusive_ptr<c10::GeneratorImpl>(
      at::xpu::detail::getDefaultXPUGenerator(device_index));
}

inline std::pair<uint64_t, uint64_t>
philoxXPUStateBridge(c10::GeneratorImpl* gen, uint64_t increment) {
  auto* xpu_gen = static_cast<at::XPUGeneratorImpl*>(gen);
  auto state = xpu_gen->philox_xpu_state(increment);
  auto [seed, offset] = at::xpu::philox::unpack(state);
  return {seed, offset};
}

} // namespace c10::xpu
#endif
