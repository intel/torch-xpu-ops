#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <aten/sycl/SummaryOpsKernel.h>

namespace at {
Tensor XPUNativeFunctions::bincount(
const Tensor& self,
const c10::optional<Tensor>& weights,
int64_t minlength){
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(common_device, self, "xpu::bincount", "self");
  c10::impl::check_and_update_common_device(common_device, weights, "xpu::bincount", "weights");
  
  return native::xpu::bincount_kernel(self,weights,minlength);
}

} // namespace at