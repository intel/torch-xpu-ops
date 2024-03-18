#include <ATen/native/CPUFallback.h>

namespace at {

static bool DEBUG_XPU_FALLBACK = false;

static void xpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  if (!DEBUG_XPU_FALLBACK) {
    TORCH_WARN_ONCE(
        "Aten Op fallback from XPU to CPU happends.",
        " This may have performance implications.",
        " If need debug the fallback ops please set environment variable `PYTORCH_DEBUG_XPU_FALLBACK=1` ");
  } else {
    TORCH_WARN(
        "The operator '",
        op.schema().operator_name(),
        "on the XPU backend and will fall back to run on the CPU.");
  }

  // TODO: do Profiling if profiler.isCPUFallbackProfilingEnabled()
  native::cpu_fallback(op, stack);
}

static void xpu_error_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "The operator '",
      op.schema().operator_name(),
      "' is not currently implemented ",
      "for the XPU device. If you want this op to be added in priority during the prototype ",
      "phase of this feature, please open issue on https://github.com/intel/torch-xpu-ops/issues. ",
      "As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_XPU_FALLBACK=1` ",
      "to use the CPU as a fallback for this op. WARNING: this will be slower than running natively ",
      "on XPU.");
}

static void xpu_force_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {}

TORCH_LIBRARY_IMPL(_, XPU, m) {
  static const char* fallback_env = getenv("PYTORCH_ENABLE_XPU_FALLBACK");
  bool enable_xpu_fallback = true;
  if (fallback_env && std::stoi(fallback_env) == 0) {
    enable_xpu_fallback = false;
  }
  if (!enable_xpu_fallback) {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&xpu_error_fallback>());
  } else {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&xpu_fallback>());
  }

  static const char* fallback_debug_env = getenv("PYTORCH_DEBUG_XPU_FALLBACK");
  if (!fallback_debug_env || std::stoi(fallback_debug_env) == 0) {
    DEBUG_XPU_FALLBACK = false;
  } else {
    DEBUG_XPU_FALLBACK = true;
  }
}

} // namespace at
