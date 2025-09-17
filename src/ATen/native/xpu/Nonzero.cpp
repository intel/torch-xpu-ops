#include <ATen/core/Tensor.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/full.h>
#include <ATen/xpu/EmptyTensor.h>
#include <ATen/native/xpu/sycl/NonzeroKernel.h>
#include <comm/TensorInfo.h>

namespace at {
namespace native {

void nonzero_common_checks(const Tensor& self, Tensor& out, const std::string& op_name) {
  TORCH_CHECK(
      self.numel() < std::numeric_limits<int>::max(),
      op_name, " is not supported for tensors with more than INT_MAX elements, \
      See https://github.com/pytorch/pytorch/issues/51871");
  TORCH_CHECK(
      self.device() == out.device(),
      "expected self and out to be on the same device, but got out on ",
      out.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      self.dim() <= XPU_MAX_TENSORINFO_DIMS,
      op_name, " is not supported for tensor with more than ",
      XPU_MAX_TENSORINFO_DIMS,
      " dimensions");
}

Tensor& nonzero_out_xpu(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      out.dtype() == at::kLong,
      "Expected object of scalar type ",
      at::kLong,
      " as out, but got ",
      out.dtype());
  nonzero_common_checks(self, out, "nonzero");
  if (self.numel() == 0) {
    out = at::detail::empty_xpu({0, self.dim()}, out.options());
    return out;
  }
  xpu::nonzero_kernel(self, out);
  return out;
}

Tensor nonzero_xpu(const Tensor& self) {
  Tensor out = at::detail::empty_xpu({0}, self.options().dtype(kLong));
  nonzero_out_xpu(self, out);
  return out;
}

Tensor& nonzero_static_out_xpu(
    const Tensor& self,
    int64_t size,
    int64_t fill_value,
    Tensor& out) {
  TORCH_CHECK(
    size >= 0, "nonzero_static: 'size' must be an non-negative integer");
  TORCH_CHECK(
    out.dtype() == at::kLong,
    "nonzero_static: Expected out tensor to have scalar type Long");
  nonzero_common_checks(self, out, "nonzero_static");
  if (self.numel() == 0) {
    out = at::full({size, self.dim()}, fill_value, out.options());
    return out;
  }

  Tensor nonzero_out = at::detail::empty_xpu({0}, self.options().dtype(kLong));
  xpu::nonzero_kernel(self, nonzero_out);
  auto nonzero_size = nonzero_out.size(0);
  out.resize_({size, self.dim()});

  if (nonzero_size > size) {
    out.copy_(nonzero_out.narrow(0, 0, size));
  } else if (nonzero_size < size) {
    auto padding = at::full({size - nonzero_size, self.dim()}, fill_value, out.options());
    out.copy_(at::cat({nonzero_out, padding}, 0));
  } else {
    out.copy_(nonzero_out);
  }
  return out;
}

Tensor nonzero_static_xpu(
    const Tensor& self,
    int64_t size,
    int64_t fill_value) {
  TORCH_CHECK(
    size >= 0, "nonzero_static: 'size' must be an non-negative integer");
  Tensor out = at::detail::empty_xpu({size, self.dim()}, self.options().dtype(kLong));
  nonzero_static_out_xpu(self, size, fill_value, out);
  return out;
}

} // namespace native
} // namespace at
