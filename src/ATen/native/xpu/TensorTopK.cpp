#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/TensorTopKKernel.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {

void topk_meta(
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");
  int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);
  TORCH_CHECK(k >= 0 && k <= sliceSize, "k not in range for dimension");

  // Build the output size, which is the dim being selected set to
  // size k
  DimVector topKSize(self.sizes().vec());
  if (!topKSize.empty()) {
    topKSize[dim] = k;
  }

  if (values.defined()) {
    at::xpu::resize_out(values, topKSize, {}, self.options());
  } else {
    values = at::xpu::create_out(topKSize, {}, self.options());
  }

  if (indices.defined()) {
    at::xpu::resize_out(indices, topKSize, {}, self.options().dtype(at::kLong));
  } else {
    indices =
        at::xpu::create_out(topKSize, {}, self.options().dtype(at::kLong));
  }
}

void topk_out_impl(
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
      "selected index k out of range");

  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
  } else {
    native::xpu::topk_kernel(self, k, dim, largest, sorted, values, indices);
  }
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::topk(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  Tensor values, indices;
  topk_meta(self, k, dim, largest, sorted, values, indices);
  topk_out_impl(self, k, dim, largest, sorted, values, indices);
  return std::tuple<Tensor, Tensor>(values, indices);
}

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::topk_out(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  topk_meta(self, k, dim, largest, sorted, values, indices);
  topk_out_impl(self, k, dim, largest, sorted, values, indices);
  return std::forward_as_tuple(values, indices);
}

} // namespace at
