#include <hal/XPUHal.h>
#include <c10/util/Exception.h>

namespace xpu_hal {
namespace {

GetDefaultGeneratorFn g_get_default_generator = nullptr;
PhiloxStateFn g_philox_state = nullptr;

} // anonymous namespace

void registerXPUGeneratorBridge(
    GetDefaultGeneratorFn get_gen,
    PhiloxStateFn philox) {
  g_get_default_generator = get_gen;
  g_philox_state = philox;
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

} // namespace xpu_hal
