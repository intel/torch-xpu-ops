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
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/IndexingKernel.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>

#include <ATen/ops/index_add_meta.h>
#include <ATen/xpu/ops/index_add_native.h>

namespace at {

namespace native {
// REGISTER_XPU_DISPATCH(masked_fill_stub, xpu::masked_fill_kernel);

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
} // namespace native

void index_func_meta_impl(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    c10::string_view func) {
  auto numel = index.numel();

  TORCH_CHECK_INDEX(
      index.dim() <= 1,
      func,
      "_(): Index is supposed to be a vector, but got dim: ",
      index.dim(),
      " with type: ",
      index.scalar_type(),
      " and size: ",
      index.sizes());
  TORCH_CHECK(
      index.scalar_type() == ScalarType::Long ||
          index.scalar_type() == ScalarType::Int,
      func,
      "_(): Expected dtype int32/int64 for index but got: ",
      index.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      func,
      "_(): self (",
      self.scalar_type(),
      ") and source (",
      source.scalar_type(),
      ") must have the same scalar type");
  TORCH_CHECK(
      dim == 0 || dim < source.dim(),
      func,
      "_(): Indexing dim ",
      dim,
      " is out of bounds of the source tensor with dim ",
      source.dim());
  TORCH_CHECK(
      numel == (source.dim() == 0 ? 1 : source.size(dim)),
      func,
      "_(): Number of indices (",
      numel,
      ") should be equal to source.size(dim): (",
      source.size(dim),
      "), for dim: ",
      dim);

  auto self_sizes = self.sizes().vec();
  auto source_sizes = source.sizes().vec();
  if (source.dim() != 0 && self.dim() != 0) {
    self_sizes.erase(self_sizes.begin() + dim);
    source_sizes.erase(source_sizes.begin() + dim);
  }
  TORCH_CHECK(
      self_sizes == source_sizes,
      "source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = ",
      self.sizes(),
      " source.shape = ",
      source.sizes());

  bool is_defined = result.defined();

  // set_output_raw_strided
  auto options = self.options();
  auto sizes = self.sizes();
  if (is_defined) {
    at::xpu::resize_out(result, sizes, {}, options);
  } else {
    result = at::xpu::create_out(sizes, {}, options);
  }

  if (is_defined) {
    at::assert_no_internal_overlap(result);
    at::assert_no_overlap(result, index);
    at::assert_no_overlap(result, source);
  }

  // A hack to run TensorIterator checks in the meta function.
  // See comment:
  // https://github.com/pytorch/pytorch/pull/65993#discussion_r760307417
  // TODO: (@krshrimali) Try inheriting from TensorIteratorBase instead.
  if (result.device() == kMeta && result.dim() > 0) {
    auto selfSlice = result.select(dim, 0);
    auto sourceSlice = source.select(dim, 0);
    auto iter =
        TensorIterator::borrowing_binary_op(selfSlice, selfSlice, sourceSlice);
  }
}

} // namespace at
