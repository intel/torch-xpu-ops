#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/TensorTopKKernel.h>

#include <comm/RegisterUtils.h>

#include <xpu/ATen/ops/topk_native.h>

namespace at {

namespace native {
TORCH_IMPL_FUNC(topk_out_xpu)
(const Tensor& self,
 int64_t k,
 int64_t dim_,
 bool largest,
 bool sorted,
 const Tensor& values,
 const Tensor& indices) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");

  // If k is 0 the result is an empty tensor, so we don't need to launch a
  // kernel.
  if (k == 0) {
    return;
  }

  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
  } else {
    native::xpu::topk_kernel(self, k, dim, largest, sorted, values, indices);
  }
}
} // namespace native

} // namespace at
