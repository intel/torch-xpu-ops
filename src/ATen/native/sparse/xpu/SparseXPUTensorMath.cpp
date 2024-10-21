// Basic Math functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/ops/add_native.h>
#include <torch/library.h>

#include <ATen/native/sparse/xpu/sycl/SparseTensorMathKernel.h>

namespace at::native::xpu {
SparseTensor& add_out_sparse_xpu(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_) {
  if (!t.is_sparse()) {
    native::xpu::add_out_dense_sparse_kernel(r_, t, src, value);
    return r_;
  }
  native::xpu::add_out_sparse_kernel(t, src, value, r_);
  return r_;
}

TORCH_LIBRARY_IMPL(aten, SparseXPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::add_.Tensor"),
      TORCH_FN(at::native::add_sparse_));
  m.impl(TORCH_SELECTIVE_NAME("aten::add.out"), TORCH_FN(add_out_sparse_xpu));
}

} // namespace at::native::xpu
