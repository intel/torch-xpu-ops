// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>

#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#include <ATen/ops/_values_native.h>

namespace at::native::xpu {

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

int64_t _nnz(const Tensor& self) {
  return at::native::_nnz_sparse(self);
}

Tensor _values(const Tensor& self) {
  return at::native::_values_sparse(self);
}

TORCH_LIBRARY_IMPL(aten, SparseXPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_sparse_coo_tensor_with_dims_and_tensors"),
      TORCH_FN(_sparse_coo_tensor_with_dims_and_tensors));
  m.impl(TORCH_SELECTIVE_NAME("aten::_nnz"), TORCH_FN(_nnz));
  m.impl(TORCH_SELECTIVE_NAME("aten::_values"), TORCH_FN(_values));
}

} // namespace at::native::xpu
