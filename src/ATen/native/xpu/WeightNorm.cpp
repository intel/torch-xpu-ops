#include <ATen/native/xpu/sycl/WeightNormKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
namespace at {
std::tuple<Tensor, Tensor> XPUNativeFunctions::_weight_norm_interface(
    const Tensor& v,
    const Tensor& g,
    int64_t dim) {
  return native::xpu::_weight_norm_interface_kernel(v, g, dim);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::_weight_norm_interface_backward(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norms,
    int64_t dim) {
  TORCH_CHECK(saved_v.is_contiguous(), "saved_v must be contiguous");
  TORCH_CHECK(saved_g.is_contiguous(), "saved_g must be contiguous");
  TORCH_CHECK(saved_norms.is_contiguous(), "saved_norms must be contiguous");
  TORCH_CHECK(
      dim == 0 || dim == saved_v.dim() - 1,
      "fused kernels can only be applied for first or last dim")

  return native::xpu::_weight_norm_interface_backward_kernel(
      grad_w, saved_v, saved_g, saved_norms, dim);
}
} // namespace at