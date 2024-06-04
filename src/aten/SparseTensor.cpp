// Basic functions on sparse tensors
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>

#include <ATen/SparseXPUNativeFunctions.h>
#include <ATen/native/SparseTensorUtils.h>

namespace at {

using namespace at::sparse;

SparseTensor new_sparse(
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  AT_ASSERT(layout.has_value() && *layout == kSparse);
  AT_ASSERT(device_or_default(device).is_xpu());
  DispatchKey dispatch_key;
  dispatch_key = DispatchKey::SparseXPU;
  return detail::make_tensor<SparseTensorImpl>(
      DispatchKeySet(dispatch_key),
      scalarTypeToTypeMeta(dtype_or_default(dtype)));
}

SparseTensor SparseXPUNativeFunctions::_sparse_coo_tensor_with_dims_and_tensors(
    int64_t sparse_dim,
    int64_t dense_dim,
    IntArrayRef size,
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
  SparseTensor self = new_sparse(dtype, layout, device, pin_memory);
  auto impl = get_sparse_impl(self);
  impl->resize_(sparse_dim, dense_dim, size);
  // NOTE: There is no guarantee that `indices` and `values` don't contain
  // AutogradMeta. However, we want to maintain the invariant that `indices_`
  // and `values_` of a sparse tensor don't contain AutogradMeta, and to achieve
  // that we shallow-copy `indices` and `values` here.
  auto indices_shallow_copy =
      Tensor(indices.unsafeGetTensorImpl()->shallow_copy_and_detach(
          /*version_counter=*/indices.unsafeGetTensorImpl()->version_counter(),
          /*allow_tensor_metadata_change=*/true));
  auto values_shallow_copy =
      Tensor(values.unsafeGetTensorImpl()->shallow_copy_and_detach(
          /*version_counter=*/values.unsafeGetTensorImpl()->version_counter(),
          /*allow_tensor_metadata_change=*/true));
  alias_into_sparse(self, indices_shallow_copy, values_shallow_copy);
  // alias_into_sparse overrides coalesced flag, so resetting the flag to
  // the desired state here:
  if (is_coalesced.has_value()) {
    impl->set_coalesced(*is_coalesced);
  }
  // TODO: alias_into_sparse sets the coalesce flag to
  // `self._values().shape[0] < 2`. There exist methods (e.g. permute
  // on COO tensors when `dims[0] != 0` holds) that force coalesced
  // flag to false even when nnz is less that 2. Here we cannot
  // determine if this is the intention of such methods but it is
  // likely that these methods are overly restrictive when estimating
  // is_coalesced state. The condition `!is_coalesced && self._nnz() <
  // 2` provides a way to detect and optimize such methods with
  // respect to estimating the is_coalesced state.
  return self;
}

} // namespace at