#include <ATen/ATen.h>
#include <ATen/core/op_registration/adaption.h>

#include <ATen/XPUNativeFunctions.h>
#include <aten/sycl/EmbeddingKernels.h>

namespace at {

Tensor XPUNativeFunctions::embedding_dense_backward(
    const Tensor& grad_output,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "embedding_dense_backward_xpu",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, indices, "embedding_dense_backward_xpu", "indices");
  return native::xpu::embedding_dense_backward_kernel(
      grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  ;
}

} // namespace at
