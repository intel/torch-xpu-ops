#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorNames.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/xpu/sycl/ShapeKernels.h>
#include <ATen/native/xpu/sycl/TensorShapeKernels.h>
#include <comm/RegisterUtils.h>
#include <xpu/ATen/ops/as_strided_copy_native.h>
#include <xpu/ATen/ops/as_strided_native.h>
#include <xpu/ATen/ops/cat_native.h>

namespace at {

namespace native {

TORCH_API at::Tensor as_strided_qtensorimpl(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    ::std::optional<int64_t> storage_offset);

Tensor as_strided_xpu(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset = c10::nullopt) {
  if (self.is_quantized()) {
    return at::native::as_strided_qtensorimpl(
        self, size, stride, storage_offset);
  }
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

TORCH_IMPL_FUNC(cat_out_xpu)
(const ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 MemoryFormat memory_format,
 const Tensor& result) {
  if (result.numel() == 0) {
    return;
  }

  xpu::cat_out_kernel(
      tensors,
      dim,
      valid,
      all_contiguous,
      all_same_dtype,
      all_same_sizes_and_stride,
      memory_format,
      result);
}

void split_with_sizes_copy_out_xpu(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim,
    TensorList out) {
  xpu::split_with_sizes_copy_out_xpu_kernel(self, split_sizes, dim, out);
}

Tensor _chunk_cat_xpu(TensorList tensors, int64_t dim, int64_t num_chunks) {
  return xpu::_chunk_cat_xpu_kernel(tensors, dim, num_chunks);
}

Tensor& _chunk_cat_out_xpu(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out) {
  return xpu::_chunk_cat_out_xpu_kernel(tensors, dim, num_chunks, out);
}

} // namespace native
} // namespace at
