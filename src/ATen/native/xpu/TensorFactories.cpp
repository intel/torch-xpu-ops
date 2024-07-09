#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <c10/xpu/XPUFunctions.h>

#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_strided_native.h>
#include <xpu/ATen/ops/_efficientzerotensor_native.h>

#include <ATen/native/xpu/sycl/ComplexKernels.h>
#include <ATen/native/xpu/sycl/RandpermKernel.h>
#include <ATen/xpu/EmptyTensor.h>

namespace at {

namespace native {

REGISTER_XPU_DISPATCH(complex_stub, &xpu::complex_kernel);

Tensor empty_xpu(
    IntArrayRef size,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  Tensor result = at::detail::empty_xpu(
      size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      memory_format_opt);
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(result);
  }
  return result;
}

Tensor empty_strided_xpu(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt,
    c10::optional<Device> device_opt,
    c10::optional<bool> pin_memory_opt) {
  Tensor result = at::detail::empty_strided_xpu(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_empty_deterministic_(result);
  }
  return result;
}

Tensor& randperm_out_xpu(
    int64_t n,
    c10::optional<Generator> generator,
    Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  at::native::check_supported_max_int_with_precision(n, result);
  result.resize_({n});

  if (n == 0) {
    return result;
  }
  native::xpu::randperm_kernel(result, n, generator);
  return result;
}

Tensor _efficientzerotensor_xpu(
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  auto device_ = device_or_default(device);
  if (!device_.has_index()) {
    device_.set_index(c10::xpu::current_device());
  }
  auto allocator = at::native::ZeroTensorAllocator(device_);
  auto dtype_ = dtype_or_default(dtype);
  auto zero_ks = at::DispatchKeySet(c10::DispatchKey::XPU) |
      at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
  auto out = at::detail::empty_generic(
      size, &allocator, zero_ks, dtype_, c10::nullopt);
  return out;
}

static void complex_check_floating(const Tensor& a, const Tensor& b) {
  TORCH_CHECK(
      (a.scalar_type() == kFloat || a.scalar_type() == kDouble ||
       a.scalar_type() == kHalf) &&
          (b.scalar_type() == kFloat || b.scalar_type() == kDouble ||
           b.scalar_type() == kHalf),
      "Expected both inputs to be Half, Float or Double tensors but got ",
      a.scalar_type(),
      " and ",
      b.scalar_type());
}

static void complex_check_dtype(
    const Tensor& result,
    const Tensor& a,
    const Tensor& b) {
  complex_check_floating(a, b);
  TORCH_CHECK(
      a.scalar_type() == b.scalar_type(),
      "Expected object of scalar type ",
      a.scalar_type(),
      " but got scalar type ",
      b.scalar_type(),
      " for second argument");
  TORCH_CHECK(
      result.scalar_type() == toComplexType(a.scalar_type()),
      "Expected object of scalar type ",
      toComplexType(a.scalar_type()),
      " but got scalar type ",
      result.scalar_type(),
      " for argument 'out'");
}

} // namespace native
} // namespace at
