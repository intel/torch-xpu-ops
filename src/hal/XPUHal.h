#pragma once

#include <c10/core/GeneratorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <utility>

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
// pytorch's PhiloxXpuState.h into kernel DLLs.  During graph capture
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
// empty_xpu() and getCurrentDeviceProperties() through xpu_hal.dll
// instead of linking torch_xpu.dll directly.
// o p a q u e   f u n c t i o n   p o i n t e r s   a v o i d
// pulling ATen headers into every kernel DLL through this header.
XPU_HAL_API void registerTorchXpuBridge(
    void* empty_xpu_primary,
    void* get_device_props);

} // namespace xpu_hal
