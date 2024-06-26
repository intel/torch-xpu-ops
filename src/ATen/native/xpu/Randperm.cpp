#include <ATen/ATen.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/xpu/sycl/RandpermKernel.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {
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
} // namespace at