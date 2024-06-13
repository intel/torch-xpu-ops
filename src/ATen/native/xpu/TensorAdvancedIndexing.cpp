#include <ATen/ATen.h>
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
// #include <ATen/xpu/XPUNativeFunctions.h>
#include <ATen/native/xpu/sycl/IndexingKernel.h>
#include <comm/RegisterUtils.h>
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

// Tensor index_select_xpu_(const Tensor& self, int64_t dim, const Tensor&
// index) {
//   Tensor result = at::empty({0}, self.options());
//   return at::native::index_select_out_xpu(self, dim, index, result);
// }

} // namespace native

// Tensor& XPUNativeFunctions::masked_fill_(
//     Tensor& self,
//     const Tensor& mask,
//     const Scalar& value) {
//   TORCH_CHECK(
//       self.device() == mask.device(),
//       "expected self and mask to be on the same device, but got mask on ",
//       mask.device(),
//       " and self on ",
//       self.device());
//   TORCH_CHECK(
//       mask.scalar_type() == kBool,
//       "masked_fill only supports boolean masks, but got dtype ",
//       mask.scalar_type());
//   auto maybe_outnames =
//       namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
//   if (at::has_internal_overlap(self) == MemOverlap::Yes) {
//     TORCH_WARN(
//         "Use of masked_fill_ on expanded tensors is deprecated. "
//         "Please clone() the tensor before performing this operation. "
//         "This also applies to advanced indexing e.g. tensor[mask] = scalar");
//   }
//   at::assert_no_partial_overlap(self, mask);

//   c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask,
//   "masked_fill_");

//   auto iter = TensorIteratorConfig()
//                   .set_check_mem_overlap(false)
//                   .check_all_same_dtype(false)
//                   .resize_outputs(false)
//                   .add_output(self)
//                   .add_const_input(self)
//                   .add_const_input(*b_mask)
//                   .build();

//   native::xpu::masked_fill_kernel(iter, value);
//   namedinference::propagate_names_if_nonempty(self, maybe_outnames);
//   return self;
// }

// Tensor& XPUNativeFunctions::masked_fill_(
//     Tensor& self,
//     const Tensor& mask,
//     const Tensor& value) {
//   TORCH_CHECK(
//       value.dim() == 0,
//       "masked_fill_ only supports a 0-dimensional value tensor, but got
//       tensor " "with ", value.dim(), " dimension(s).");
//   // We hit this function if either of the input tensor lives on XPU.
//   // It is ok, if `value` is `CPU` tensor but we should not allow `self` or
//   // `mask` to be CPU tensor. Check for `self` and `mask` being on same
//   device
//   // exists in `masked_fill_` (Scalar version).
//   TORCH_CHECK(
//       self.device().is_xpu(),
//       "masked_fill_: Expected inputs to be on same device")
//   return XPUNativeFunctions::masked_fill_(self, mask, value.item());
// }

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
  at::xpu::resize_out(result, sizes, {}, options);
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

// Tensor& XPUNativeFunctions::index_add_out(
//     const Tensor& self,
//     int64_t dim,
//     const Tensor& index,
//     const Tensor& source,
//     const Scalar& alpha,
//     Tensor& out) {
//   std::optional<Device> common_device = std::nullopt;
//   c10::impl::check_and_update_common_device(
//       common_device, self, "xpu::index_add_out", "self");
//   c10::impl::check_and_update_common_device(
//       common_device, index, "xpu::index_add_out", "index");
//   c10::impl::check_and_update_common_device(
//       common_device, source, "xpu::index_add_out", "source");
//   dim = maybe_wrap_dim(dim, self.dim());
//   index_func_meta_impl(out, self, dim, index, source, "index_add");
//   native::xpu::index_add_kernel(self, dim, index, source, alpha, out);
//   return out;
// }

} // namespace at
