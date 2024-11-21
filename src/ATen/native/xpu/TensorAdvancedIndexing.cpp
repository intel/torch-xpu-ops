#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/ReductionType.h>
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
REGISTER_XPU_DISPATCH(index_copy_stub, &xpu::index_copy_kernel);
REGISTER_XPU_DISPATCH(put_stub, &xpu::put_kernel);
REGISTER_XPU_DISPATCH(take_stub, &xpu::take_kernel);

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

TORCH_IMPL_FUNC(index_reduce_xpu_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const c10::string_view reduce,
 bool include_self,
 const Tensor& result){

    std::optional<Device> common_device = std::nullopt;
    c10::impl::check_and_update_common_device(
        common_device, self, "xpu::index_add_out", "self");
    c10::impl::check_and_update_common_device(
        common_device, index, "xpu::index_add_out", "index");
    c10::impl::check_and_update_common_device(
        common_device, source, "xpu::index_add_out", "source");
    dim = maybe_wrap_dim(dim, self.dim());

    int reduce_type = 0; 
    reduce == "prod"? reduce_type = 1;
    reduce == "mean"? reduce_type = 2;
    reduce == "amax"? reduce_type = 3;
    reduce == "amin"? reduce_type = 4;

    switch(reduce_type){
        case 0: //invalid 
            TORCH_CHECK(false, "reduce argument must be one of the following choices: prod, mean, amax or amin. The choice was ", reduce, ".");
        case 1: //prod
            //index_reduce_kernel(self, dim, index, source, include_self, ReductionType::PROD, reduce_multiply, result);
            index_reduce_kernel(self, dim, index, source, include_self, ReductionType::PROD, result);
            break;
        case 2: //mean
            index_reduce_kernel(self, dim, index, source, include_self, ReductionType::MEAN, result);
            auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
            counts.index_add_(dim, index, at::ones_like(source));
            counts.masked_fill_(counts == 0, 1);     
            if (result.is_floating_point() || result.is_complex()) {
                result.div_(counts);
            } else {
                result.div_(counts, "floor");
            }       
            break;
        case 3: //amax
            index_reduce_kernel(self, dim, index, source, include_self, ReductionType::MAX, result);
            break;
        case 4: //amin
            index_reduce_kernel(self, dim, index, source, include_self, ReductionType::MIN, result);
            break;
    }
}

} // namespace native
} // namespace at
