#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/RepeatKernel.h>
#include <ATen/xpu/XPUNativeFunctions.h>
namespace at {
Tensor XPUNativeFunctions::repeat_interleave(
    const Tensor& repeats,
    c10::optional<int64_t> output_size) {
  return at::native::xpu::repeat_interleave_kernel(repeats, output_size);
}
} // namespace at
