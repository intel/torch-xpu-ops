#pragma once

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor _sparse_csr_sum_xpu(
  const Tensor& input,
  IntArrayRef dims_to_sum,
  bool keepdim,
  std::optional<ScalarType> dtype);

TORCH_XPU_API Tensor _sparse_csr_prod_xpu(
  const Tensor& input,
  IntArrayRef dims_to_reduce,
  bool keepdim,
  std::optional<ScalarType> dtype);

} // namespace at::native::xpu