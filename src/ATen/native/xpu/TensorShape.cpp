#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorNames.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/xpu/sycl/ShapeKernels.h>
#include <comm/RegisterUtils.h>

#include <ATen/xpu/ops/as_strided_copy_native.h>
#include <ATen/xpu/ops/as_strided_native.h>
#include <ATen/xpu/ops/cat_native.h>

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

} // namespace native
} // namespace at
