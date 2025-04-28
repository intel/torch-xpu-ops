#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/SummaryOpsKernels.h>
#include <comm/SYCLContext.h>

#include <xpu/ATen/ops/bincount_native.h>

namespace at {
namespace native {

Tensor _bincount_xpu(
    const Tensor& self,
    const std::optional<Tensor>& weights_opt,
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

Tensor _histc_xpu(
    const Tensor& self,
    int64_t nbins,
    const Scalar& min,
    const Scalar& max) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("_histc_xpu");
  return native::xpu::_histc_kernel(self, nbins, min, max);
}

Tensor& _histc_out_xpu(
    const Tensor& self,
    int64_t bins,
    const Scalar& min,
    const Scalar& max,
    Tensor& result) {
  auto ret = _histc_xpu(self, bins, min, max);
  at::native::resize_output(result, ret.sizes());
  result.copy_(ret);
  return result;
}

} // namespace native
} // namespace at
