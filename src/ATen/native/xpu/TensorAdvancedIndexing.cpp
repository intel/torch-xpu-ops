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
REGISTER_XPU_DISPATCH(index_fill_stub, &xpu::index_fill_kernel);

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

Tensor count_nonzero_xpu(const Tensor& self, IntArrayRef dims) {
  return (self != 0).sum(dims);
}

Tensor index_copy_meta(
    const Tensor& self,
    int64_t& dim,
    const Tensor& index,
    const Tensor& source,
    Tensor& out) {
  dim = maybe_wrap_dim(dim, self.dim());

  Tensor& result = out;

  // Memory overlap checks need to be done after resizing (if required) is done.
  // But it only makes sense to do these checks when result was defined, hence
  // the boolean variable `check_result` here.
  // For more details, see:
  // https://github.com/pytorch/pytorch/pull/63312#discussion_r694794832 and
  // https://github.com/pytorch/pytorch/issues/63837
  bool check_result = result.defined();
  TensorIterator iter = TensorIteratorConfig()
                            .add_output(result)
                            .add_borrowed_const_input(self)
                            .build();
  iter.set_output_raw_strided(
      0,
      self.sizes(),
      {},
      self.options(),
      result.has_names() ? result.names() : DimnameList{});
  if (check_result) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, index);
    at::assert_no_overlap(result, source);
  }

  TORCH_CHECK_INDEX(
      index.dim() < 2,
      "index_copy_(): Index should have dimension 1 or 0 (got ",
      index.dim(),
      ")");

  int64_t numIndices = index.numel();
  if (source.dim() == 0 && numIndices != 1) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source is scalar, index should have one element (got ",
        numIndices,
        ")");
  } else if (
      (source.dim() != self.dim()) && (source.dim() != 0 && self.dim() != 0)) {
    TORCH_CHECK_INDEX(
        false,
        "index_copy_(): When source and destination are not scalars, their dimensionality must match. Source dimensionality (",
        source.dim(),
        "), destination dimensionality (",
        self.dim(),
        ")");
  }

  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long,
      "index_copy_(): Expected a long tensor for index, but got ",
      index.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "index_copy_(): self and source expected to have the same dtype, but got (self) ",
      self.scalar_type(),
      " and (source) ",
      source.scalar_type());
  TORCH_CHECK(
      self.device() == source.device() && self.device() == index.device(),
      "index_copy_(): self, index and source expected to be in the same device, but got (self) ",
      self.device(),
      ", (index) ",
      index.device(),
      ", and (source) ",
      source.device());

  // Check that source and destination slices have the same size
  auto selfSlicedSizes = self.sizes().vec();
  if (!selfSlicedSizes.empty()) {
    selfSlicedSizes.erase(selfSlicedSizes.begin() + dim);
  }
  auto sourceSlicedSizes = source.sizes().vec();
  if (!sourceSlicedSizes.empty()) {
    sourceSlicedSizes.erase(sourceSlicedSizes.begin() + dim);
  }
  if (selfSlicedSizes.size() != sourceSlicedSizes.size() ||
      !std::equal(
          selfSlicedSizes.begin(),
          selfSlicedSizes.end(),
          sourceSlicedSizes.begin())) {
    std::stringstream ss;
    ss << "index_copy_(): Source/destination tensor must have same slice shapes. ";
    ss << "Destination slice shape: " << selfSlicedSizes << " at dimension "
       << dim;
    ss << " and source slice shape: " << sourceSlicedSizes
       << " at dimension 0.";
    TORCH_CHECK(false, ss.str());
  }
  TORCH_CHECK_INDEX(
      source.dim() == 0 || numIndices == source.size(dim),
      "index_copy_(): Number of indices (",
      numIndices,
      ") should be equal to source.size(dim) (",
      source.size(dim),
      ")");

  return iter.output();
}

Tensor XPUNativeFunctions::index_copy(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  Tensor out;
  return XPUNativeFunctions::index_copy_out(self, dim, index, source, out);
}

Tensor& XPUNativeFunctions::index_copy_(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source) {
  return XPUNativeFunctions::index_copy_out(self, dim, index, source, self);
}

Tensor& XPUNativeFunctions::index_copy_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    Tensor& out) {
  out = index_copy_meta(self, dim, index, source, out);

  if (!out.is_same(self))
    out.copy_(self);

  if (index.numel() == 0) {
    return out;
  }

  // See Note [Enabling Deterministic Operations]
  if (globalContext().deterministicAlgorithms()) {
    torch::List<std::optional<Tensor>> indices;
    indices.reserve(dim + 1);
    for (const auto i : c10::irange(dim)) {
      (void)i;
      indices.emplace_back();
    }
    indices.emplace_back(index);
    out.index_put_(indices, source, false);
    return out;
  }

  native::xpu::index_copy_kernel(dim, index, source, out);
  return out;
}

} // namespace native
} // namespace at
