#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/sycl/ScanUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

#include <ATen/native/xpu/sycl/CumprodKernel.h>
#include <ATen/native/xpu/sycl/CumsumKernel.h>

namespace at::native::xpu {

static c10::MaybeOwned<Tensor> contiguous_out_arg(const Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
  return c10::MaybeOwned<Tensor>::owned(
      at::empty(tensor.sizes(), tensor.options()));
}

void cumsum_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto result_ = contiguous_out_arg(result);

  launch_cumsum_kernel(*result_, self, dim);

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

void cumprod_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto result_ = contiguous_out_arg(result);

  launch_cumprod_kernel(*result_, self, dim);

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

} // namespace at::native::xpu
