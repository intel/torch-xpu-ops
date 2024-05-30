#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <aten/sycl/BatchNormKernels.h>

namespace at {

::std::tuple<Tensor, Tensor> XPUNativeFunctions::batch_norm_stats(
    const Tensor& input,
    double eps) {
  return native::xpu::batch_norm_stats_kernel(input, eps);
}

} // namespace at
