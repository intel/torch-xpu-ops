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
  static const char* enable_xpu_fallback =
      getenv("PYTORCH_ENABLE_XPU_FALLBACK");
  if (!enable_xpu_fallback || std::stoi(enable_xpu_fallback) == 0) {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&xpu_error_fallback>());
  } else {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&xpu_fallback>());
  }

  static const char* debug_xpu_fallback = getenv("PYTORCH_DEBUG_XPU_FALLBACK");
  if (!debug_xpu_fallback || std::stoi(enable_xpu_fallback) == 0) {
    DEBUG_XPU_FALLBACK = false;
  } else {
    DEBUG_XPU_FALLBACK = true;
  }
}

/*
 * TODO: Move the following registration to the end of all XPU aten op
 * registrations
 */

/*
 * Register fallback to CPU for ops specified in env variable
 * "PYTORCH_XPU_FALLBACK_OP" eg. export
 * PYTORCH_XPU_FALLBACK_OP=abs.out,div.Scalar,div.Tensor,div_.Scalar,div_.Tensor
 */
/*
TORCH_LIBRARY_IMPL(aten, XPU, m) {
  static const char* fallback_op_str = getenv("PYTORCH_XPU_FALLBACK_OP");
  if (!fallback_op_str) {
    return;
  }
  std::istringstream iss(fallback_op_str);
  std::string op_name;
  while (std::getline(iss, op_name, ',')) {
    TORCH_WARN(
        "The operator '", op_name, "' will be forced to fallback to CPU.");
    m.impl(
        op_name.c_str(),
        torch::CppFunction::makeFromBoxedFunction<&xpu_fallback>());
  }
}
*/

} // namespace at
