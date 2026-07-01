#pragma once

#include <c10/core/GeneratorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <utility>

#ifndef _WIN32
#include <ATen/xpu/XPUGeneratorImpl.h>
#endif

// On Windows, xpu_hal is a STATIC library WHOLE_ARCHIVE'd into torch_xpu.dll.
// Kernel DLLs import symbols from torch_xpu.dll via the import lib.
// On Linux, everything links into one .so -- no bridge needed. The header
// provides inline forwarding to pytorch APIs directly.

#ifdef _WIN32
#ifdef XPU_HAL_BUILD
#define XPU_HAL_API __declspec(dllexport)
#elif defined(XPU_HAL_IMPORT)
#define XPU_HAL_API __declspec(dllimport)
#pragma comment(lib, "torch_xpu.lib")
#else
#define XPU_HAL_API
#endif
#else
#define XPU_HAL_API
#endif

namespace xpu_hal {

using GetDefaultGeneratorFn =
    c10::intrusive_ptr<c10::GeneratorImpl> (*)(int64_t device_index);
using PhiloxStateFn = std::pair<uint64_t, uint64_t> (*)(
    c10::GeneratorImpl* gen,
    uint64_t increment);

// Mirrors PhiloxXpuState layout without pulling pytorch's header into
// kernel DLLs on Windows.
struct PhiloxCaptureState {
  union Payload {
    uint64_t val;
    int64_t* ptr;
    Payload() : val(0) {}
  };
  Payload seed_{};
  Payload offset_{};
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};
using PhiloxCaptureStateFn =
    PhiloxCaptureState (*)(c10::GeneratorImpl* gen, uint64_t increment);

#ifdef _WIN32

// Called once by torch_xpu during XPU generator initialization.
XPU_HAL_API void registerXPUGeneratorBridge(
    GetDefaultGeneratorFn get_gen,
    PhiloxStateFn philox);
XPU_HAL_API void registerXPUGeneratorCaptureBridge(
    PhiloxCaptureStateFn capture_fn);

// Bridge accessors for kernel DLLs.
XPU_HAL_API c10::intrusive_ptr<c10::GeneratorImpl> getDefaultGenerator(
    int64_t device_index);

XPU_HAL_API std::pair<uint64_t, uint64_t> philoxState(
    c10::GeneratorImpl* gen,
    uint64_t increment);

XPU_HAL_API PhiloxCaptureState
philoxCaptureState(c10::GeneratorImpl* gen, uint64_t increment);

XPU_HAL_API void registerTorchXpuBridge(
    void* empty_xpu_primary,
    void* get_device_props);

#else // Linux: inline forwarding to pytorch APIs (no DLL boundary)

inline void registerXPUGeneratorBridge(GetDefaultGeneratorFn, PhiloxStateFn) {}
inline void registerXPUGeneratorCaptureBridge(PhiloxCaptureStateFn) {}
inline void registerTorchXpuBridge(void*, void*) {}

inline c10::intrusive_ptr<c10::GeneratorImpl> getDefaultGenerator(
    int64_t device_index) {
  return at::xpu::detail::getDefaultXPUGenerator(device_index)
      .getIntrusivePtr();
}

inline std::pair<uint64_t, uint64_t> philoxState(
    c10::GeneratorImpl* gen,
    uint64_t increment) {
  auto* xpu_gen = static_cast<at::XPUGeneratorImpl*>(gen);
  return xpu_gen->philox_engine_inputs(increment);
}

inline PhiloxCaptureState philoxCaptureState(
    c10::GeneratorImpl* gen,
    uint64_t increment) {
  auto* xpu_gen = static_cast<at::XPUGeneratorImpl*>(gen);
  auto state = xpu_gen->philox_xpu_state(increment);
  PhiloxCaptureState result;
  result.seed_.ptr = state.seed_.ptr;
  result.offset_.ptr = state.offset_.ptr;
  result.offset_intragraph_ = state.offset_intragraph_;
  result.captured_ = state.captured_;
  return result;
}

#endif // _WIN32

} // namespace xpu_hal
