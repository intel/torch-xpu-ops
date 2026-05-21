#include <ATen/core/class_type.h>
#include <ATen/core/ivalue.h>
#include <torch/library.h>

namespace at::native::xpu {
namespace {

void queue_push_boxed(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet ks,
    c10::Stack* stack) {
  (void)op;
  (void)ks;

  auto x = torch::jit::pop(*stack).toTensor();
  auto tq = torch::jit::pop(*stack);

  TORCH_CHECK(
      tq.isObject(),
      "Expected _TorchScriptTesting::queue_push first argument to be a TorchBind object");

  auto class_type = tq.type()->cast<c10::ClassType>();
  TORCH_CHECK(
      class_type != nullptr,
      "Expected _TorchScriptTesting::queue_push first argument to have ClassType");
  TORCH_CHECK(
      class_type->hasMethod("push"),
      "Expected _TorchScriptTesting::queue_push object to implement method 'push'");

  // Match CPU implementation in test_custom_class_registrations.cpp.
  class_type->getMethod("push")({std::move(tq), x.clone()});
}

} // namespace

TORCH_LIBRARY_IMPL(_TorchScriptTesting, XPU, m) {
  m.impl(
      "queue_push",
      torch::CppFunction::makeFromBoxedFunction<&queue_push_boxed>());
}

TORCH_LIBRARY_IMPL(_TorchScriptTesting, AutocastXPU, m) {
    m.impl(
            "queue_push",
            torch::CppFunction::makeFromBoxedFunction<&queue_push_boxed>());
}

} // namespace at::native::xpu
