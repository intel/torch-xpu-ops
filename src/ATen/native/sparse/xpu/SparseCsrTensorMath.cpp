#pragma once
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>

namespace at::native{

using namespace at::sparse;

Tensor _sparse_csr_sum_xpu(
  const Tensor& input,
  IntArrayRef dims_to_sum,
  bool keepdim,
  std::optional<ScalarType> dtype) {
    std::cout<< "---_sparse_csr_sum_xpu---" << std::endl;
    return xpu::_sparse_csr_sum_xpu_kernel(input, dims_to_sum, keepdim, dtype);
}

Tensor _sparse_csr_prod_xpu(
  const Tensor& input,
  IntArrayRef dims_to_reduce,
  bool keepdim,
  std::optional<ScalarType> dtype) {
    std::cout<< "---_sparse_csr_prod_xpu---" << std::endl;
    return xpu::_sparse_csr_prod_xpu_kernel(input, dims_to_reduce, keepdim, dtype);
}

} // namespace at::native