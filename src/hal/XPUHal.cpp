#include <hal/XPUHal.h>
#include <c10/util/Exception.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Layout.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <ATen/core/TensorBase.h>
#include <c10/xpu/XPUDeviceProp.h>

namespace xpu_hal {
namespace {

GetDefaultGeneratorFn g_get_default_generator = nullptr;
PhiloxStateFn g_philox_state = nullptr;
PhiloxCaptureStateFn g_philox_capture_state = nullptr;

// Typed function pointers for torch_xpu symbols bridged through xpu_hal.
using EmptyXpuPrimaryFn = at::TensorBase (*)(
    c10::IntArrayRef size,
    c10::ScalarType dtype,
    c10::optional<c10::Device> device_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);
using GetDevicePropFn = c10::xpu::DeviceProp* (*)();

EmptyXpuPrimaryFn g_empty_xpu_primary = nullptr;
GetDevicePropFn g_get_device_props = nullptr;

// Accessors used by the wrapper functions in at::detail / at::xpu.
EmptyXpuPrimaryFn getEmptyXpuPrimaryFn() { return g_empty_xpu_primary; }
GetDevicePropFn getDevicePropFn() { return g_get_device_props; }

} // anonymous namespace

void registerXPUGeneratorBridge(
    GetDefaultGeneratorFn get_gen,
    PhiloxStateFn philox) {
  g_get_default_generator = get_gen;
  g_philox_state = philox;
}

void registerXPUGeneratorCaptureBridge(
    PhiloxCaptureStateFn capture_fn) {
  g_philox_capture_state = capture_fn;
}

void registerTorchXpuBridge(
    void* empty_xpu_primary,
    void* get_device_props) {
  g_empty_xpu_primary = reinterpret_cast<EmptyXpuPrimaryFn>(empty_xpu_primary);
  g_get_device_props = reinterpret_cast<GetDevicePropFn>(get_device_props);
}

c10::intrusive_ptr<c10::GeneratorImpl> getDefaultGenerator(
    int64_t device_index) {
  TORCH_CHECK(
      g_get_default_generator != nullptr,
      "XPU generator bridge not registered. "
      "Ensure torch_xpu.dll is loaded before calling XPU generator functions.");
  return g_get_default_generator(device_index);
}

std::pair<uint64_t, uint64_t> philoxState(
    c10::GeneratorImpl* gen,
    uint64_t increment) {
  TORCH_CHECK(
      g_philox_state != nullptr,
      "XPU generator bridge not registered. "
      "Ensure torch_xpu.dll is loaded before calling XPU generator functions.");
  return g_philox_state(gen, increment);
}

PhiloxCaptureState philoxCaptureState(
    c10::GeneratorImpl* gen,
    uint64_t increment) {
  TORCH_CHECK(
      g_philox_capture_state != nullptr,
      "XPU generator capture bridge not registered. "
      "Ensure torch_xpu.dll is loaded before calling XPU generator functions.");
  return g_philox_capture_state(gen, increment);
}

} // namespace xpu_hal

#ifdef XPU_HAL_EMIT_ALIAS_WRAPPERS
// ---------------------------------------------------------------------------
// Wrapper functions that kernel DLLs import from xpu_hal.dll.
// These match the mangled names of torch_xpu.dll exports so the linker
// resolves kernel DLL references without /FORCE:UNRESOLVED.
// ---------------------------------------------------------------------------

// XPUGeneratorImpl::device_type() — used by Distribution kernels and
// Foreach kernels via check_generator<XPUGeneratorImpl>() in SYCL code.
// Minimal class definition (matches only the static method; the full
// definition lives in torch_xpu.dll).
namespace at {
struct XPUGeneratorImpl {
  static c10::DeviceType device_type();
};
c10::DeviceType XPUGeneratorImpl::device_type() {
  return c10::DeviceType::XPU;
}
} // namespace at

// empty_xpu — used by Foreach, Nonzero, Randperm, Rrelu, and
// TensorFactories kernels when creating output tensors.
namespace at::detail {

TensorBase empty_xpu(
    c10::IntArrayRef size,
    c10::ScalarType dtype,
    c10::optional<c10::Device> device_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto fn = xpu_hal::getEmptyXpuPrimaryFn();
  TORCH_CHECK(
      fn != nullptr,
      "empty_xpu bridge not registered. "
      "Ensure torch_xpu.dll is loaded before calling empty_xpu.");
  return fn(size, dtype, device_opt, memory_format_opt);
}

TensorBase empty_xpu(
    c10::IntArrayRef size,
    c10::optional<c10::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(
      !layout_opt.has_value() || layout_opt.value() == c10::Layout::Strided,
      "empty_xpu only supports Strided layout.");
  const auto dtype =
      dtype_opt.has_value() ? dtype_opt.value() : c10::ScalarType::Float;
  return at::detail::empty_xpu(size, dtype, device_opt, memory_format_opt);
}

TensorBase empty_xpu(
    c10::IntArrayRef size,
    const c10::TensorOptions& options) {
  // c10::TensorOptions stores dtype as caffe2::TypeMeta; convert to
  // optional<ScalarType> for the underlying bridge call.
  auto dtype_opt = options.dtype_opt();
  c10::optional<c10::ScalarType> scalar_type_opt = c10::nullopt;
  if (dtype_opt.has_value()) {
    scalar_type_opt = c10::typeMetaToScalarType(dtype_opt.value());
  }
  return at::detail::empty_xpu(
      size,
      scalar_type_opt,
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

} // namespace at::detail

// getCurrentDeviceProperties — used by SoftMaxKernels for occupancy tuning.
namespace at::xpu {
c10::xpu::DeviceProp* getCurrentDeviceProperties() {
  auto fn = xpu_hal::getDevicePropFn();
  TORCH_CHECK(
      fn != nullptr,
      "getCurrentDeviceProperties bridge not registered. "
      "Ensure torch_xpu.dll is loaded before querying device properties.");
  return fn();
}
} // namespace at::xpu
#endif
