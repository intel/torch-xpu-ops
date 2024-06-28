#include <ATen/native/xpu/sycl/SummaryOpsKernels.h>
#include <comm/SYCLContext.h>

#include <ATen/xpu/ops/bincount_native.h>
namespace at {
namespace native {
Tensor _bincount_xpu(
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
} // namespace native

} // namespace at
