#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/utils/ParamUtils.h>
#include <aten/sycl/DistanceKernel.h>

namespace at {

Tensor XPUNativeFunctions::_cdist_forward(const Tensor& x1, const Tensor& x2, const double p, c10::optional<int64_t> compute_mode) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(common_device, x1, "xpu::_cdist_forward", "x1");
  c10::impl::check_and_update_common_device(common_device, x2, "xpu::_cdist_forward", "x2");

  TORCH_CHECK(
      x1.dim() >= 2,
      "cdist only supports at least 2D tensors, X1 got: ",
      x1.dim(),
      "D");
  TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
  TORCH_CHECK(x1.size(-1) == x2.size(-1), "X1 and X2 must have the same number of columns. X1: ", x1.size(-1), " X2: ", x2.size(-1));
  
  return native::xpu::cdist_impl(x1,x2,p,compute_mode);
}

} // namespace at