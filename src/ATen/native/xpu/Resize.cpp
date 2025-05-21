#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <c10/core/Allocator.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>

#include <ATen/native/Resize.h>
#include <xpu/ATen/ops/copy.h>
#include <xpu/ATen/ops/resize_native.h>
#include <xpu/ATen/ops/set_native.h>

#include <ATen/native/xpu/sycl/ResizeKernel.h>

namespace at {

namespace native {
const at::Tensor& resize_(
    const at::Tensor& self,
    at::IntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt);
}
namespace native::xpu {

const Tensor& resize_xpu_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  int64_t old_storage_nbytes =
      self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  resize_impl_xpu_(self_, size, /*strides=*/std::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(
          at::globalContext().deterministicAlgorithms() &&
          at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
  }
  return self;
}

const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    std::optional<MemoryFormat> optional_memory_format = std::nullopt) {
  return resize_xpu_(self, the_template.sizes(), optional_memory_format);
}

Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  // Dispatch explicitly to bypass redispatching in ATen CPU fallback routine.
  if (dst.is_xpu()) {
    resize_xpu_(dst, self.sizes(), std::nullopt);
  } else {
    at::native::resize_(dst, self.sizes());
  }
  return const_cast<Tensor&>(dst.copy_(self, false));
}

Tensor _copy_from(const Tensor& self, const Tensor& dst, bool non_blocking) {
  dst.resize_as_(self);
  return const_cast<Tensor&>(dst.copy_(self, non_blocking));
}

// Should not register the operator. Desc of resize_as_ and
// _copy_from_and_resize native_function.yaml is simplistic since PyTorch
// intends backend should not register it (e.g. CPU/CUDA) or handle
// sanity check by backend (e.g. MPS).
TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("aten::_copy_from_and_resize"),
      TORCH_FN(_copy_from_and_resize));
  m.impl(TORCH_SELECTIVE_NAME("aten::_copy_from"), TORCH_FN(_copy_from));
}
} // namespace native::xpu

namespace native {

const at::Tensor& resize_xpu_(
    const at::Tensor& self,
    at::IntArrayRef size,
    std::optional<at::MemoryFormat> memory_format) {
  return native::xpu::resize_xpu_(self, size, memory_format);
}

Tensor& set_storage_xpu_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  at::native::checkSetStorage(self, source, storage_offset, size, stride);

  self.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  std::optional<IntArrayRef> stride_opt = stride.data() != nullptr
      ? std::optional<IntArrayRef>(stride)
      : std::nullopt;
  native::xpu::resize_impl_xpu_(self.unsafeGetTensorImpl(), size, stride_opt);
  return self;
}

Tensor& set_xpu_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(Storage::use_byte_size_t(), 0, c10::GetAllocator(kXPU), true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}
} // namespace native
} // namespace at
