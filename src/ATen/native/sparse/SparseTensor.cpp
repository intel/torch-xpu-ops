// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>

#include <ATen/ops/_values_native.h>
#include <ATen/ops/_coalesced_native.h>
#include <ATen/ops/_indices_native.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_native.h>
#include <ATen/ops/copy_sparse_to_sparse.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/is_coalesced_native.h>
#include <ATen/ops/sparse_dim_native.h>
#endif

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

TORCH_LIBRARY_IMPL(aten, SparseXPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_sparse_coo_tensor_with_dims_and_tensors"),
      TORCH_FN(_sparse_coo_tensor_with_dims_and_tensors));
  m.impl(TORCH_SELECTIVE_NAME("aten::_nnz"), TORCH_FN(at::native::_nnz_sparse));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_values"),
      TORCH_FN(at::native::_values_sparse));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_indices"),
      TORCH_FN(at::native::_indices_sparse));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::is_coalesced"),
      TORCH_FN(at::native::is_coalesced_sparse));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::sparse_dim"),
      TORCH_FN(at::native::sparse_dim_sparse));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::dense_dim"),
      TORCH_FN(at::native::dense_dim_sparse));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::copy_sparse_to_sparse_"),
      TORCH_FN(at::copy_sparse_to_sparse_));
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_coalesced_"),
      TORCH_FN(at::native::_coalesced_sparse_));
}

} // namespace at::native::xpu
