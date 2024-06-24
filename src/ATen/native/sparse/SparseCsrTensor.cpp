// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nnz_native.h>
#endif

namespace at::native::xpu {

int64_t _nnz_csr(const Tensor& self) {
  return at::native::_nnz_sparse_csr(self);
}

TORCH_LIBRARY_IMPL(aten, SparseCsrXPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_nnz"), TORCH_FN(_nnz_csr));
}

} // namespace at::native::xpu
