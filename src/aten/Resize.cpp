#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ResizeCommon.h>
#include <torch/library.h>

#include <aten/Copy.h>
#include <aten/sycl/CopyKernel.h>

#include <comm/SYCLContext.h>
#include <comm/XPUGuard.h>

namespace at {
namespace native {
namespace xpu {

namespace impl {

void resize_bytes_xpu(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(storage->resizable(),
      "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(allocator != nullptr,
      "Trying to resize storage without an allocator");

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
        data.get(),
        storage->data(),
        std::min(storage->nbytes(), size_bytes));
  }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

static inline
void maybe_resize_storage_xpu(TensorImpl* self, size_t new_size_bytes) {
  // It does not make sense to try to resize a storage
  // to hold 0 elements, and this can break
  // if storage_offset is positive but
  // new_size is 0, so just bail in that case
  // (same comment is in Resize.h)
  if (self->numel() == 0) {
    return;
  }

  const Storage &storage = self->unsafe_storage();
  TORCH_CHECK(storage, "Tensor: invalid null storage");
  if (new_size_bytes > storage.nbytes()) {
    resize_bytes_xpu(storage.unsafeGetStorageImpl(), new_size_bytes);
  }
}

inline TensorImpl* resize_impl_xpu_(
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
  int64_t old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  impl::resize_impl_xpu_(self_, size, /*strides=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms()
      && at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
  }
  return self;
}

} // impl

const at::Tensor & resize_(
    const at::Tensor & self,
    at::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format) {
  return impl::resize_xpu_(self, size, memory_format);
}

Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  return copy_xpu(const_cast<Tensor&>(dst), self, false);
}

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::resize_"), TORCH_FN(resize_));
  m.impl(TORCH_SELECTIVE_NAME("aten::_copy_from_and_resize"), TORCH_FN(_copy_from_and_resize));
}

}}} // at::native::xpu
