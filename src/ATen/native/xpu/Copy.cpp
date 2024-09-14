#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/CachingHostAllocator.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUEvent.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <ATen/xpu/detail/XPUHooks.h>
#include <c10/core/ScalarType.h>
#include <c10/xpu/XPUStream.h>

#include <ATen/native/xpu/sycl/CopyKernel.h>
#include <ATen/native/xpu/sycl/UnaryComplexKernels.h>
#include <comm/SYCLContext.h>
#include <comm/XPUGuard.h>

using namespace at;
using namespace at::xpu;

namespace at {
namespace native::xpu {

static bool copy_requires_temporaries(TensorIterator& iter, bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GPU.
    TORCH_INTERNAL_ASSERT(dst_device.is_xpu() && src_device.is_xpu());
    return false;
  } else if (
      dst_device.is_xpu() && src_device.is_xpu() &&
      (dst_device != src_device)) {
    // Across device copies need temporaries if p2p not enabled
    return !p2p_enabled;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use memcpyAsync
    return false;
  } else if (dst_device.is_xpu() && src_device.is_xpu()) {
    // Copies between GPUs can use the copy kernel if P2P is supported
    return !p2p_enabled;
  } else {
    // The remaining cases require temporaries. For example, this includes
    // non-contiguous copies between CPU and GPU.
    return true;
  }
}

static bool maybe_enable_p2p_access(Device dst_device, Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }

  auto dst_queue = getCurrentXPUStream(dst_device.index()).queue();
  auto src_queue = getCurrentXPUStream(src_device.index()).queue();
  auto dst_dev = dst_queue.get_device();
  auto src_dev = src_queue.get_device();
  return src_dev.ext_oneapi_can_access_peer(
      dst_dev, sycl::ext::oneapi::peer_access::access_supported);
}

void memcpyAsync(
    TensorIterator& iter,
    XPUStream& copy_stream,
    bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);
  if (dst_device == src_device) {
    copy_kernel(iter);
  } else {
    TORCH_INTERNAL_ASSERT(p2p_enabled == true);
    auto dst = (char*)iter.data_ptr(0);
    auto src = (char*)iter.data_ptr(1);
    size_t size = iter.numel() * iter.element_size(0);
    auto q = copy_stream.queue();
    q.copy(src, dst, size);
  }
}

void copy_device_to_device(
    TensorIterator& iter,
    bool non_blocking,
    bool p2p_enabled) {
  auto numel = iter.numel();
  if (numel == 0) {
    return;
  }

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
  bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();
  bool memcpy_eligible =
      same_type && same_conj && same_neg && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  c10::DeviceGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy.
  XPUStream copy_stream = getCurrentXPUStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    XPUEvent dst_ready;
    device_guard.set_index(dst_device.index());
    dst_ready.record(getCurrentXPUStream(dst_device.index()));

    device_guard.set_index(src_device.index());
    dst_ready.block(copy_stream);
  }

  if (memcpy_eligible) {
    // SYCL queue.memcpy performance is worse than SYCL copy kernel
    // implementation. JIRA:
    // https://jira.devtools.intel.com/browse/CMPLRLLVM-41292
    memcpyAsync(iter, copy_stream, p2p_enabled);
  } else {
    if (same_neg) {
      if (!same_conj) {
        conj_kernel(iter);
      } else {
        copy_kernel(iter);
      }
    } else {
      if (!same_conj) {
        neg_conj_kernel(iter);
      } else {
        neg_kernel(iter);
      }
    }
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.
    // Still on src_device, record stream event
    XPUEvent src_ready;
    src_ready.record(copy_stream);

    device_guard.set_index(dst_device.index());
    src_ready.block(getCurrentXPUStream(dst_device.index()));
  }
}

void _copy_xpu(TensorIterator& iter, bool non_blocking) {
  AT_ASSERT(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // Enable p2p access between devices. (No-op if it invovles the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  if (copy_requires_temporaries(iter, p2p_enabled)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    Tensor dst_contig;
    Tensor src_contig;

    bool requires_across_device_temporaries =
        (iter.device(0) != iter.device(1)) &&
        iter.device(0).type() == c10::DeviceType::XPU &&
        iter.device(1).type() == c10::DeviceType::XPU;

    // If non_blocking is true - type conversions are performed on the GPU
    // for CPU-GPU copies, otherwise type conversions are performed on the CPU.
    // Type conversions are performed on the src device for GPU-GPU copies.
    if (iter.device_type(0) == kXPU || non_blocking) {
      if (requires_across_device_temporaries) {
        dst_contig = at::empty(dst.sizes(), dst.options().device(kCPU));
      } else {
        dst_contig = dst.is_contiguous()
            ? dst
            : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      }

      // TODO: Support quantization
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();

    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type)
          ? dst
          : at::empty_like(dst, iter.dtype(1), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // propagate the correct conjugate bit
    dst_contig._set_conj(dst.is_conj());
    src_contig._set_conj(iter.tensor(1).is_conj());

    dst_contig._set_neg(dst.is_neg());
    src_contig._set_neg(iter.tensor(1).is_neg());

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      // TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

  // Copy on GPU (or between GPUs)
  if (dst_device.is_xpu() && src_device.is_xpu()) {
    copy_device_to_device(iter, non_blocking, p2p_enabled);
    return;
  }

  // Copy between CPU and GPU
  c10::xpu::OptionalXPUGuard device_guard;
  enum { _H2D_, _D2H_ } copy_kind;
  if (dst_device.type() == c10::DeviceType::XPU && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    copy_kind = _H2D_;
  } else if (dst_device.is_cpu() && src_device.type() == c10::DeviceType::XPU) {
    device_guard.set_device(src_device);
    copy_kind = _D2H_;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);

  auto q = getCurrentSYCLQueue();
  if (non_blocking) {
    if (copy_kind == _H2D_) {
      if (at::detail::getXPUHooks().isPinnedPtr(src)) {
        q.memcpy(dst, src, nbytes);
        at::xpu::CachingHostAllocator_recordEvent(
            const_cast<void*>(src),
            iter.tensor(1).storage().data_ptr().get_context(),
            at::xpu::getCurrentXPUStream());
      } else {
        // Using stage memory for async copy to avoid incorrect
        // free of src host memory before async H2D copy. E.g. memory allocated
        // by CPU tensor factory won't be cached in CPU allocator. When host
        // memory is freed with CPU tensor dtor at the end of train main loop,
        // but the corresponding H2D copy might not have been executed yet.
        auto stage_mem_dptr = at::xpu::HostAlloc(nbytes);
        void* stage_mem = stage_mem_dptr.get();
        if (!stage_mem) {
          throw std::runtime_error(
              "Fail to allocate host memory from XPU HostAllocator");
        }

        std::memcpy(stage_mem, src, nbytes);
        q.memcpy(dst, stage_mem, nbytes);
        at::xpu::CachingHostAllocator_recordEvent(
            const_cast<void*>(stage_mem),
            stage_mem_dptr.get_context(),
            at::xpu::getCurrentXPUStream());
      }
    } else {
      q.memcpy(dst, src, nbytes);
      if (at::detail::getXPUHooks().isPinnedPtr(dst)) {
        at::xpu::CachingHostAllocator_recordEvent(
            const_cast<void*>(dst),
            iter.tensor(0).storage().data_ptr().get_context(),
            at::xpu::getCurrentXPUStream());
      }
    }
  } else {
    auto e = q.memcpy(dst, src, nbytes);
    e.wait();
  }

  if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
    iter.tensor(0).conj_physical_();
  }
  if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
    iter.tensor(0).neg_();
  }
}

} // namespace native::xpu

Tensor& XPUNativeFunctions::copy_(
    Tensor& self,
    const Tensor& src,
    bool non_blocking) {
  if (self._is_zerotensor()) {
    TORCH_CHECK(
        false,
        "ZeroTensors are immutable. Please materialize the tensor using `.clone()`, if you want a mutable zero tensor.");
  }
  if (src._is_zerotensor()) {
    return self.zero_();
  }

  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  if (self.is_same(src)) {
    return self;
  }

  // TODO: Support quantization

  // Exit early if self and src are views of the same data
  const bool is_same_data =
      (self.is_alias_of(src) && self.storage_offset() == src.storage_offset() &&
       self.strides().equals(src.strides()) &&
       self.sizes().equals(src.sizes()) &&
       self.scalar_type() == src.scalar_type() &&
       self.is_conj() == src.is_conj() && self.is_neg() == src.is_neg());
  if (is_same_data) {
    return self;
  }

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(true)
                  .add_output(self)
                  .add_input(src)
                  .resize_outputs(false)
                  .check_all_same_dtype(false)
                  .check_all_same_device(false)
                  .build();

  if (iter.numel() == 0) {
    return self;
  }

  native::xpu::_copy_xpu(iter, non_blocking);

  return self;
}

Tensor XPUNativeFunctions::_to_copy(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    bool non_blocking,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  return at::native::_to_copy(
      self,
      dtype,
      layout,
      device,
      pin_memory,
      non_blocking,
      optional_memory_format);
}
} // namespace at
