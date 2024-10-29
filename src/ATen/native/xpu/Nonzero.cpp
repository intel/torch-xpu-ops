#include <ATen/core/Tensor.h>
#include <ATen/xpu/EmptyTensor.h>

#include <ATen/native/xpu/sycl/NonzeroKernel.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>

namespace at {
namespace native {
Tensor& nonzero_out_xpu(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      self.numel() < std::numeric_limits<int>::max(),
      "nonzero is not supported for tensors with more than INT_MAX elements, \
  See https://github.com/pytorch/pytorch/issues/51871");
  TORCH_CHECK(
      out.dtype() == at::kLong,
      "Expected object of scalar type ",
      at::kLong,
      " as out, but got ",
      out.dtype());
  TORCH_CHECK(
      self.device() == out.device(),
      "expected self and out to be on the same device, but got out on ",
      out.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      self.dim() <= MAX_DIMS,
      "nonzero is not supported for tensor with more than ",
      MAX_DIMS,
      " dimensions");

  xpu::nonzero_kernel(self, out);
  return out;
}

Tensor nonzero_xpu(const Tensor& self) {
  Tensor out = at::detail::empty_xpu({0}, self.options().dtype(kLong));
  nonzero_out_xpu(self, out);
  return out;
}
} // namespace native
} // namespace at