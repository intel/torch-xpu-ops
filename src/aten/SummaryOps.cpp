#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <aten/sycl/SummaryOpsKernel.h>
#include <comm/SYCLContext.h>
#include <comm/Runtime.h>

namespace at {
Tensor XPUNativeFunctions::bincount(
const Tensor& self,
const c10::optional<Tensor>& weights_opt,
int64_t minlength){
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(common_device, self, "xpu::bincount", "self");
  c10::impl::check_and_update_common_device(common_device, weights_opt, "xpu::bincount", "weights_opt");
  
  c10::MaybeOwned<Tensor> weights_maybe_owned = at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  if (weights_opt.has_value()) {
      // See Note [Writing Nondeterministic Operations]
      // Nondeterministic if weights are given, because of floating point
      // atomicAdd usage
      globalContext().alertNotDeterministic("_bincount_xpu");
  }


  return native::xpu::bincount_kernel(self,weights,minlength);
}

} // namespace at