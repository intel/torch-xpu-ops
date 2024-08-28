#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <c10/core/Allocator.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/set_native.h>
#endif

#include <ATen/native/xpu/sycl/ResizeKernel.h>

namespace at {
namespace native::xpu {

const Tensor& resize_xpu_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  int64_t old_storage_nbytes =
      self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  resize_impl_xpu_(self_, size, /*strides=*/c10::nullopt);
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
    c10::optional<MemoryFormat> optional_memory_format = c10::nullopt) {
  return resize_(self, the_template.sizes(), optional_memory_format);
}

Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  // Dispatch explicitly to bypass redispatching in ATen CPU fallback routine.
  if (dst.is_xpu()) {
    resize_xpu_(dst, self.sizes(), c10::nullopt);
  } else {
    at::native::resize_(dst, self.sizes());
  }
  return at::XPUNativeFunctions::copy_(const_cast<Tensor&>(dst), self, false);
}

// For test infrastructure
Tensor _copy_from(const Tensor& self, const Tensor& dst, bool non_blocking) {
  dst.resize_as_(self);
  return at::XPUNativeFunctions::copy_(
      const_cast<Tensor&>(dst), self, non_blocking);
}

// Should not register the operator. Desc of
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

const at::Tensor& XPUNativeFunctions::resize_(
    const at::Tensor& self,
    at::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format) {
  return native::xpu::resize_xpu_(self, size, memory_format);
}

Tensor& XPUNativeFunctions::set_(Tensor& self, Storage source) {
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  return self.set_(source, 0, new_size, {});
}

Tensor& XPUNativeFunctions::set_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  at::native::checkSetStorage(self, source, storage_offset, size, stride);

  self.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  c10::optional<IntArrayRef> stride_opt = stride.data() != nullptr
      ? c10::optional<IntArrayRef>(stride)
      : c10::nullopt;
  native::xpu::resize_impl_xpu_(self.unsafeGetTensorImpl(), size, stride_opt);
  return self;
}

Tensor& XPUNativeFunctions::set_(Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

Tensor& XPUNativeFunctions::set_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(Storage::use_byte_size_t(), 0, c10::GetAllocator(kXPU), true);
  result.set_(storage, 0, {0}, {});
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

} // namespace at
