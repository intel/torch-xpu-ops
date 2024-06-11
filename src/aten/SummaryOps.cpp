#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <aten/sycl/SummaryOpsKernels.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>

namespace at {
Tensor XPUNativeFunctions::bincount(
    const Tensor& self,
    const c10::optional<Tensor>& weights_opt,
    int64_t minlength) {
  c10::MaybeOwned<Tensor> weights_maybe_owned =
      at::borrow_from_optional_tensor(weights_opt);
  const Tensor& weights = *weights_maybe_owned;

  if (weights_opt.has_value()) {
    // See Note [Writing Nondeterministic Operations]
    // Nondeterministic if weights are given, because of floating point
    // atomicAdd usage
    globalContext().alertNotDeterministic("_bincount_xpu");
  }

  return native::xpu::bincount_kernel(self, weights, minlength);
}

} // namespace at