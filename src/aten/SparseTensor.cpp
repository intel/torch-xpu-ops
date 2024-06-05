// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#endif

namespace at {
namespace native::xpu {

Tensor _sparse_coo_tensor_with_dims_and_tensors(
    int64_t sparse_dim,
    int64_t dense_dim,
    c10::SymIntArrayRef size,
    const Tensor& indices,
    const Tensor& values,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<bool> is_coalesced) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      indices,
      "xpu::_sparse_coo_tensor_with_dims_and_tensors",
      "indices");
  c10::impl::check_and_update_common_device(
      common_device,
      values,
      "xpu::_sparse_coo_tensor_with_dims_and_tensors",
      "values");
  return at::native::new_with_dims_and_tensor_sparse_symint(
      sparse_dim,
      dense_dim,
      size,
      indices,
      values,
      dtype,
      layout,
      device,
      pin_memory,
      is_coalesced);
}

TORCH_LIBRARY_IMPL(aten, SparseXPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("_sparse_coo_tensor_with_dims_and_tensors"),
      TORCH_FN(_sparse_coo_tensor_with_dims_and_tensors));
}

} // namespace native::xpu
} // namespace at