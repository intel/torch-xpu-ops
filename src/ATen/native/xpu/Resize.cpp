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

#include <ATen/native/xpu/sycl/CopyKernel.h>
#include <comm/SYCLContext.h>
#include <comm/XPUGuard.h>

namespace at {
namespace native::xpu {

extern Tensor& _copy_xpu(Tensor& self, const Tensor& src, bool non_blocking);

void resize_bytes_xpu(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(
      storage->resizable(), "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(
      allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage->device();

  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage->set_nbytes(0);
    return;
  }

  c10::xpu::XPUGuard guard(device.index());
  at::DataPtr data = allocator->allocate(size_bytes);
  if (storage->data_ptr()) {
    at::globalContext().lazyInitXPU();
    auto q = at::xpu::getCurrentSYCLQueue();

    q.memcpy(
        data.get(), storage->data(), std::min(storage->nbytes(), size_bytes));
  }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

static inline void maybe_resize_storage_xpu(
    TensorImpl* self,
    size_t new_size_bytes) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (self->numel() == 0) {
    return;
  }

  const Storage& storage = self->unsafe_storage();
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  if (new_size_bytes > storage.nbytes()) {
    resize_bytes_xpu(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

TensorImpl* resize_impl_xpu_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;
  }

  // NB: We don't need to hold the device guard when calling from TH
  at::xpu::OptionalXPUGuard guard;
  if (device_guard) {
    guard.set_index(self->storage().device().index());
  }

  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->storage_offset();
  size_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);
  } else {
    self->set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }
  maybe_resize_storage_xpu(self, storage_size);

  return self;
}

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
  return native::xpu::_copy_xpu(const_cast<Tensor&>(dst), self, false);
}

// For test infrastructure
Tensor _copy_from(const Tensor& self, const Tensor& dst, bool non_blocking) {
  dst.resize_as_(self);
  return native::xpu::_copy_xpu(const_cast<Tensor&>(dst), self, non_blocking);
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
