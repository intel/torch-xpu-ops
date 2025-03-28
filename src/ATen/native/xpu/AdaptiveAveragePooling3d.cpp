#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/xpu/sycl/AdaptiveAveragePooling3dKernels.h>

#include <ATen/ops/adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/adaptive_avg_pool3d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>

namespace at::native {

Tensor& adaptive_avg_pool3d_out_xpu(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  at::native::xpu::adaptive_avg_pool3d_kernel(output, input, output_size);
  return output;
}

Tensor adaptive_avg_pool3d_xpu(const Tensor& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  at::native::xpu::adaptive_avg_pool3d_kernel(output, input, output_size);
  return output;
}

Tensor& adaptive_avg_pool3d_backward_out_xpu(
    const Tensor& gradOutput_,
    const Tensor& input,
    Tensor& gradInput) {
  globalContext().alertNotDeterministic("adaptive_avg_pool3d_backward_xpu");
  at::native::xpu::adaptive_avg_pool3d_backward_kernel(
      gradInput, gradOutput_, input);
  return gradInput;
}

Tensor adaptive_avg_pool3d_backward_xpu(
    const Tensor& gradOutput_,
    const Tensor& input) {
  globalContext().alertNotDeterministic("adaptive_avg_pool3d_backward_xpu");
  auto gradInput = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::xpu::adaptive_avg_pool3d_backward_kernel(
      gradInput, gradOutput_, input);
  return gradInput;
}

} // namespace at::native
