#include <dlfcn.h>
#include <ATen/ceil_div.h>
#include <c10/XPU/XPUGuard.h>

#include <ishmem_extension.hpp>
#include <xccl/XPUSymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <ATen/ceil_div.h>

#include <ishmem.h>
#include <ishmemx.h>

namespace c10d::ishmem_extension {

extern "C" void ishmem_init() __attribute__((weak));

// Check if iSHMEM is available
bool is_ishmem_available() {
  // Runtime check
  static std::mutex mutex;
  static int is_available = -2;
  std::lock_guard<std::mutex> lock(mutex);

  // Checked if the symbol is statically linked
  if(is_available == -2 && ishmem_init) {
    is_available = 1;
  }

  if (is_available == -2) {
    void* handle{};
    // Open the shared library, RTLD_LAZY defers symbol resolution until needed
    handle = dlopen("libishmem_host.so.3", RTLD_LAZY);
    if (!handle) {
      std::cerr << dlerror() << "\n";
      is_available = 0;
    } else {
      is_available = 1;
      // Close the shared library
      dlclose(handle);
    }
  }
  return is_available == 1;
}

void ishmem_put(at::Tensor& tensor, const int64_t peer) {
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "put op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
  auto rank = hdl->get_rank();
  void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
  auto buffer_size = tensor.numel() * tensor.element_size();
  TORCH_CHECK(peer < hdl->get_world_size(), "peer must be smaller than world size");

  c10::xpu::XPUGuard guard(tensor.device());
  auto sycl_queue = at::xpu::getCurrentXPUStream().queue();
  ishmemx_putmem_on_stream(buffer_ptr, tensor.data_ptr(), buffer_size, peer, sycl_queue);
}

void ishmem_wait_for_signal(at::Tensor& sigpad, int64_t signal, int64_t peer) {
  c10::xpu::XPUGuard guard(sigpad.device());
  auto sycl_queue = at::xpu::getCurrentXPUStream().queue();
  ishmemx_signal_wait_until_on_stream(static_cast<uint64_t*>(sigpad.data_ptr()), ISHMEM_CMP_EQ, signal, sycl_queue);
}

void ishmem_put_with_signal(at::Tensor& tensor, at::Tensor& sigpad, int64_t signal, int64_t peer) {
  auto buffer_size = tensor.numel() * tensor.element_size();

  c10::xpu::XPUGuard guard(tensor.device());
  auto sycl_queue = at::xpu::getCurrentXPUStream().queue();
  ishmemx_putmem_signal_on_stream(
    tensor.mutable_data_ptr(),
    tensor.mutable_data_ptr(),
    buffer_size,
    static_cast<uint64_t*>(sigpad.mutable_data_ptr()),
    signal,
    ISHMEM_SIGNAL_SET,
    peer,
    sycl_queue);
}

void ishmem_get(at::Tensor& tensor, const int64_t peer) {
  // TODO: support non-contiguous tensors
  TORCH_CHECK(tensor.is_contiguous(),
      "get op currently supports contiguous tensors only");
  // TODO: rendezvous should remember the group name
  auto hdl = c10d::symmetric_memory::rendezvous(tensor, "0");
  auto rank = hdl->get_rank();
  void* buffer_ptr = hdl->get_buffer_ptrs()[rank];
  auto buffer_size = tensor.numel() * tensor.element_size();
  TORCH_CHECK(peer < hdl->get_world_size(), "peer must be smaller than world size");

  c10::xpu::XPUGuard guard(tensor.device());
  auto sycl_queue = at::xpu::getCurrentXPUStream().queue();
  ishmemx_getmem_on_queue(tensor.mutable_data_ptr(), buffer_ptr, buffer_size, peer, sycl_queue);
}
} // namespace c10d::ishmem_extension


TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ishmem_put", c10d::ishmem_extension::ishmem_put);
  m.impl("ishmem_get", c10d::ishmem_extension::ishmem_get);
  m.impl("ishmem_wait_for_signal", c10d::ishmem_extension::ishmem_wait_for_signal);
  m.impl("ishmem_put_with_signal", c10d::ishmem_extension::ishmem_put_with_signal);
}
