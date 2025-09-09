#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include <c10/util/error.h>

#include <c10/xpu/XPUCachingAllocator.h>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <xccl/XPUSymmetricMemoryUtils.hpp>

namespace c10d::symmetric_memory {

std::string getSymmMemBackendXPU() {
  static auto val = c10::utils::get_env("TORCH_SYMMMEM");
  if (val.has_value()) {
    TORCH_CHECK(
        val.value() == "XPU",
        "TORCH_SYMMMEM environment variable must be 'XPU'.");
    return val.value();
  }
  return "XPU";
}

bool device_has_multicast_support(int device_idx) {
  return false;
}

bool allow_overlapping_devices() {
  return false;
}

void map_block(
    void** ptr,
    ze_physical_mem_handle_t handle,
    size_t size,
    int device_idx) {
  sycl::queue current_queue = at::xpu::getCurrentXPUStream().queue();
  sycl::context sycl_ctx = current_queue.get_context();
  ze_context_handle_t ze_context =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_ctx);
  // 1. Reserve virtual address space
  void* virtual_ptr = nullptr;
  ze_result_t status = zeVirtualMemReserve(
      ze_context, // context
      nullptr, // let L0 pick virtual address
      size, // size
      &virtual_ptr // out: reserved address
  );
  TORCH_CHECK(status == ZE_RESULT_SUCCESS, "zeVirtualMemReserve failed");

  // 2. Map physical memory to virtual address
  status = zeVirtualMemMap(
      ze_context,
      virtual_ptr, // virtual memory to map to
      size,
      handle, // physical memory handle
      0, // flags
      ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE // ze_memory_access_attribute_t
  );
  TORCH_CHECK(status == ZE_RESULT_SUCCESS, "zeVirtualMemMap failed");

  // 3. Set access attributes
  ze_memory_access_attribute_t access = ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE;
  status =
      zeVirtualMemSetAccessAttribute(ze_context, virtual_ptr, size, access);
  TORCH_CHECK(
      status == ZE_RESULT_SUCCESS, "zeVirtualMemSetAccessAttribute failed");

  // 4. Return pointer
  *ptr = virtual_ptr;
}

} // namespace c10d::symmetric_memory
