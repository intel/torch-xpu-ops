#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/IndexingKernels.h>
#include <ATen/native/xpu/sycl/ScatterGatherKernels.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>

#include <ATen/ops/index_add_meta.h>
#include <xpu/ATen/ops/index_add_native.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(index_stub, &xpu::index_kernel);
REGISTER_XPU_DISPATCH(index_put_stub, &xpu::index_put_kernel);
REGISTER_XPU_DISPATCH(
    index_put_with_sort_stub,
    &xpu::index_put_deterministic_kernel);
// REGISTER_XPU_DISPATCH(index_stub, &xpu::index_kernel);
REGISTER_XPU_DISPATCH(scatter_stub, &xpu::scatter_kernel);
REGISTER_XPU_DISPATCH(scatter_fill_stub, &xpu::scatter_fill_kernel);
REGISTER_XPU_DISPATCH(scatter_add_stub, &xpu::scatter_add_kernel);
REGISTER_XPU_DISPATCH(scatter_reduce_stub, &xpu::scatter_reduce_kernel);
REGISTER_XPU_DISPATCH(scatter_reduce_two_stub, &xpu::scatter_reduce_two_kernel);
REGISTER_XPU_DISPATCH(
    scatter_scalar_reduce_stub,
    &xpu::scatter_scalar_reduce_kernel);
REGISTER_XPU_DISPATCH(gather_stub, &xpu::gather_kernel);

// REGISTER_XPU_DISPATCH(index_fill_stub, &xpu::index_fill_kernel);

TORCH_IMPL_FUNC(index_add_xpu_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const Scalar& alpha,
 const Tensor& result) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::index_add_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "xpu::index_add_out", "index");
  c10::impl::check_and_update_common_device(
      common_device, source, "xpu::index_add_out", "source");
  dim = maybe_wrap_dim(dim, self.dim());
  //   index_func_meta_impl(result, self, dim, index, source, "index_add");
  native::xpu::index_add_kernel(self, dim, index, source, alpha, result);
}

Tensor& masked_fill__xpu(
    Tensor& self,
    const Tensor& mask,
    const Scalar& value) {
  TORCH_CHECK(
      self.device() == mask.device(),
      "expected self and mask to be on the same device, but got mask on ",
      mask.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      mask.scalar_type() == kBool,
      "masked_fill only supports boolean masks, but got dtype ",
      mask.scalar_type());
  auto maybe_outnames =
      namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of masked_fill_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_output(self)
                  .add_const_input(self)
                  .add_const_input(*b_mask)
                  .build();

  xpu::masked_fill_kernel(iter, value);
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor& masked_fill__xpu(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0,
      "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ",
      value.dim(),
      " dimension(s).");
  // We hit this function if either of the input tensor lives on XPU.
  // It is ok, if `value` is `CPU` tensor but we should not allow `self` or
  // `mask` to be CPU tensor. Check for `self` and `mask` being on same device
  // exists in `masked_fill_` (Scalar version).
  TORCH_CHECK(
      self.device().is_xpu(),
      "masked_fill_: Expected inputs to be on same device")
  return masked_fill__xpu(self, mask, value.item());
}

// TODO: Should reuse source in stock PyTorch when in-tree.
static bool all_strides_match(TensorList tensors) {
  TORCH_CHECK(!tensors.empty());
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

// Replace indexed dimensions in src with stride 0 and the size of the result
// tensor. The offset in these dimensions is computed by the kernel using the
// index tensor's values and the stride of src. The new shape is not meaningful.
// It's used to make the shape compatible with the result tensor.
static Tensor restride_src(
    const Tensor& src,
    int64_t dims_before,
    int64_t dims_indexed,
    IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());
  auto strides = DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(
      shape.begin() + dims_before,
      replacement_shape.begin(),
      replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

// Add dimensions of size 1 to an index tensor so that it can be broadcast to
// the result shape and iterated over element-wise like the result tensor and
// the restrided src.
static Tensor reshape_indexer(
    const Tensor& index,
    int64_t dims_before,
    int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list) {
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  IntArrayRef replacement_shape;
  for (const auto dim : c10::irange(indices_list.size())) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) !=
          indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) ==
          replacement_shape.end()) {
    TORCH_CHECK_INDEX(
        false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  if (indices.size() >= 2 && (this->src.device().type() == kXPU)) {
    if (!all_strides_match(indices)) {
      for (auto& indice : indices) {
        indice = indice.contiguous();
      }
    }
  }
}

Tensor count_nonzero_xpu(const Tensor& self, IntArrayRef dims) {
  return (self != 0).sum(dims);
}

} // namespace native
} // namespace at
