#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <c10/xpu/XPUFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/empty_strided_native.h>
#endif

#include <ATen/native/xpu/sycl/ComplexKernels.h>
#include <ATen/native/xpu/sycl/RandpermKernel.h>
#include <ATen/native/xpu/sycl/TensorFactoriesKernels.h>
#include <ATen/xpu/EmptyTensor.h>

namespace at {

Tensor& XPUNativeFunctions::eye_out(int64_t n, Tensor& result) {
  return XPUNativeFunctions::eye_out(n, n, result);
}

Tensor& XPUNativeFunctions::eye_out(int64_t n, int64_t m, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  int64_t stride = result.stride(0) + result.stride(1);

  Tensor diag = result.as_strided({sz}, {stride});
  diag.fill_(1);
  return result;
}

Tensor XPUNativeFunctions::empty(
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

Tensor XPUNativeFunctions::empty_strided(
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

Tensor XPUNativeFunctions::clone(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  return at::native::clone(self, memory_format);
}

Tensor XPUNativeFunctions::_efficientzerotensor(
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

Tensor& XPUNativeFunctions::complex_out(
    const Tensor& real,
    const Tensor& imag,
    Tensor& result) {
  complex_check_dtype(result, real, imag);
  auto iter = TensorIteratorConfig()
                  .add_output(result)
                  .add_const_input(real)
                  .add_const_input(imag)
                  .check_all_same_dtype(false)
                  .build();
  native::xpu::complex_kernel(iter);
  return result;
}

Tensor& XPUNativeFunctions::polar_out(
    const Tensor& abs,
    const Tensor& angle,
    Tensor& result) {
  complex_check_dtype(result, abs, angle);
  auto iter = TensorIteratorConfig()
                  .add_output(result)
                  .add_const_input(abs)
                  .add_const_input(angle)
                  .check_all_same_dtype(false)
                  .build();
  native::xpu::polar_kernel(iter);
  return result;
}

Tensor& XPUNativeFunctions::randperm_out(
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

Tensor XPUNativeFunctions::tril_indices(
    int64_t row,
    int64_t col,
    int64_t offset,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return native::xpu::tril_indices_kernel(row, col, offset, options);
}

Tensor XPUNativeFunctions::triu_indices(
    int64_t row,
    int64_t col,
    int64_t offset,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return native::xpu::triu_indices_kernel(row, col, offset, options);
}

} // namespace at
