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
#include <ATen/native/xpu/sycl/ScatterGatherKernels.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>

#include <ATen/ops/index_add_meta.h>
#include <ATen/xpu/ops/index_add_native.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(masked_fill_stub, xpu::masked_fill_kernel);
REGISTER_XPU_DISPATCH(index_put_stub, xpu::index_put_kernel);
REGISTER_XPU_DISPATCH(
    index_put_with_sort_stub,
    xpu::index_put_deterministic_kernel);
REGISTER_XPU_DISPATCH(index_stub, xpu::index_kernel);
REGISTER_XPU_DISPATCH(scatter_stub, xpu::scatter_kernel);
REGISTER_XPU_DISPATCH(scatter_fill_stub, xpu::scatter_fill_kernel);
REGISTER_XPU_DISPATCH(scatter_add_stub, xpu::scatter_add_kernel);
REGISTER_XPU_DISPATCH(scatter_reduce_stub, xpu::scatter_reduce_kernel);
REGISTER_XPU_DISPATCH(scatter_reduce_two_stub, xpu::scatter_reduce_two_kernel);
REGISTER_XPU_DISPATCH(
    scatter_scalar_reduce_stub,
    xpu::scatter_scalar_reduce_kernel);

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
} // namespace at
