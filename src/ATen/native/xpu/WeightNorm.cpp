#include <ATen/native/xpu/sycl/WeightNormKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include<iostream>
namespace at {
std::tuple<Tensor, Tensor> XPUNativeFunctions::_weight_norm_interface(
    const Tensor& v,
    const Tensor& g,
    int64_t dim) {
  std::cout << "zhy---line:" << __LINE__ << ", call _weight_norm_interface"
            << std::endl;
  return native::xpu::_weight_norm_interface_kernel(v, g, dim);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::_weight_norm_interface_backward(
    const Tensor& grad_w,
    const Tensor& saved_v,
    const Tensor& saved_g,
    const Tensor& saved_norms,
    int64_t dim) {
  std::cout << "zhy---line:" << __LINE__ << ", call _weight_norm_interface"
            << std::endl;
  return native::xpu::_weight_norm_interface_backward_kernel(
      grad_w, saved_v, saved_g, saved_norms, dim);
}
} // namespace at