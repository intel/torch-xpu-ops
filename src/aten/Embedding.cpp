#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <aten/sycl/Embedding.h>

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
      "xpu::embedding_dense_backward",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, indices, "xpu::embedding_dense_backward", "indices");
  return native::xpu::embedding_dense_backward(
      grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  ;
}

} // namespace at
