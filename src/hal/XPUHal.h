#pragma once

#include <c10/core/GeneratorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <utility>

#ifndef _WIN32
#include <ATen/xpu/XPUGeneratorImpl.h>
#endif

// xpu_hal is built as a standalone DLL in BUILD_SEPARATE_OPS mode.
// The dllexport/dllimport pattern is needed because:
//  - xpu_hal exports bridge symbols
//  - Kernel DLLs import them from xpu_hal.dll
//  - torch_xpu's own .obj files (e.g. XPUGeneratorImpl.cpp) see neither
//
// When XPU_HAL_BUILD is defined, symbols are dllexport.
// When XPU_HAL_IMPORT is defined (kernel DLLs), symbols are dllimport
//   and the pragma pulls in torch_xpu's import lib.
// Otherwise (torch_xpu internal consumers), symbols are plain.

#ifdef _WIN32
#ifdef XPU_HAL_BUILD
#define XPU_HAL_API __declspec(dllexport)
#elif defined(XPU_HAL_IMPORT)
#define XPU_HAL_API __declspec(dllimport)
#pragma comment(lib, "xpu_hal.lib")
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

// Extended capture state that mirrors PhiloxXpuState without pulling
// pytorch's PhiloxXpuState.h into kernel DLLs. During graph capture
// the payload is extragraph pointers; outside capture it is raw values.
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
using PhiloxCaptureStateFn = PhiloxCaptureState (*)(
    c10::GeneratorImpl* gen,
    uint64_t increment);

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

XPU_HAL_API PhiloxCaptureState philoxCaptureState(
    c10::GeneratorImpl* gen,
    uint64_t increment);

// Register torch_xpu function pointers so kernel DLLs can call
// empty_xpu(), resize_impl_xpu_(), and getCurrentDeviceProperties()
// through xpu_hal.dll
// instead of linking torch_xpu.dll directly.
XPU_HAL_API void registerTorchXpuBridge(
    void* empty_xpu_primary,
    void* resize_impl_xpu,
    void* get_device_props);

#else

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
  result.captured_ = state.captured_;
  if (state.captured_) {
    result.seed_.ptr = state.seed_.ptr;
    result.offset_.ptr = state.offset_.ptr;
    result.offset_intragraph_ = state.offset_intragraph_;
  } else {
    auto [seed, offset] = at::xpu::philox::unpack(state);
    result.seed_.val = seed;
    result.offset_.val = offset;
  }
  return result;
}

#endif

} // namespace xpu_hal
