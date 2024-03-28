#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>

#include <aten/sycl/ScanKernels.h>
// #include <ATen/native/ReduceOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at::native::xpu {

static c10::MaybeOwned<Tensor> contiguous_out_arg(const Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
  return c10::MaybeOwned<Tensor>::owned(
      at::empty(tensor.sizes(), tensor.options()));
}

void cumsum_xpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  if (self.is_floating_point() || self.is_complex()) {
    // See Note [Writing Nondeterministic Operations]
    // Issue reporting nondeterministic behavior:
    // https://github.com/pytorch/pytorch/issues/75240
    globalContext().alertNotDeterministic("cumsum_xpu_kernel");
  }
  auto result_ = contiguous_out_arg(result);
  launch_cumsum_xpu_kernel(*result_, self, dim);
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

// TODO: what's this
// REGISTER_CUDA_DISPATCH(cumsum_stub, &cumsum_cuda_kernel);

} // namespace at::native::xpu
