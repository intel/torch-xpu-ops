#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/UnfoldBackwardKernels.h>

namespace at {

Tensor XPUNativeFunctions::unfold_backward(
    const Tensor& grad,
    IntArrayRef input_sizes,
    int64_t dim,
    int64_t size,
    int64_t step) {
  auto grad_input = at::zeros(input_sizes, grad.options());
  if (step >= size) {
    auto gI_unfolded = grad_input.unfold(dim, size, step);
    gI_unfolded.copy_(grad);
    return grad_input;
  }
  native::xpu::unfold_backward_kernel(grad_input, grad, dim, size, step);

  return grad_input;
}

} // namespace at
