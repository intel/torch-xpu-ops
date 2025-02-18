#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>
#include <xpu/ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <xpu/ATen/ops/_convert_indices_from_csr_to_coo_native.h>

namespace at::native {

using namespace at::sparse;

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_xpu)
(const Tensor& input,
 const int64_t size,
 const bool out_int32,
 const Tensor& result) {
  xpu::convert_indices_from_coo_to_csr_structured_kernel(
      input, size, out_int32, result);
};

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_xpu)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose,
 const Tensor& result) {
  xpu::convert_indices_from_csr_to_coo_structured_kernel(
      crow_indices, col_indices, out_int32, transpose, result);
};

Tensor _sparse_csr_sum_xpu(
    const Tensor& input,
    IntArrayRef dims_to_sum,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  return xpu::_sparse_csr_sum_xpu_kernel(input, dims_to_sum, keepdim, dtype);
}

Tensor _sparse_csr_prod_xpu(
    const Tensor& input,
    IntArrayRef dims_to_reduce,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  return xpu::_sparse_csr_prod_xpu_kernel(
      input, dims_to_reduce, keepdim, dtype);
}

} // namespace at::native
