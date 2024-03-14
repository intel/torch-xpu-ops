#include <ATen/native/CPUFallback.h>

namespace at {

static void xpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  TORCH_WARN_ONCE(
      "The operator '",
      op.schema().operator_name(),
      "' is not currently supported ",
      "on the XPU backend and will fall back to run on the CPU.",
      " This may have performance implications.");

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

TORCH_LIBRARY_IMPL(_, XPU, m) {
  static const char* enable_xpu_fallback =
      getenv("PYTORCH_ENABLE_XPU_FALLBACK");
  if (!enable_xpu_fallback || std::stoi(enable_xpu_fallback) == 0) {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&xpu_error_fallback>());
  } else {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&xpu_fallback>());
  }
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  // These ops are not supported via XPU backend currently, and we fallback to
  // run on CPU. For the rest of unsupported ops the user needs to pass
  // 'PYTORCH_ENABLE_XPU_FALLBACK=1' to fallback on CPU, otherwise we will error
  // out. m.impl("div.Scalar",
  // torch::CppFunction::makeFromBoxedFunction<&xpu_fallback>());
}

} // namespace at
