#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/xpu/XPUPluggableAllocator.h>

namespace {
using namespace c10d::symmetric_memory;

// Alloc functor for MemPool
void* xpu_symm_alloc(size_t size, int device, sycl::queue* /*queue*/) {
  static auto allocator = get_allocator(c10::DeviceType::XPU);
  // Note: the group info is now specified at the time of rendezvous instead of
  // allocation. We thus pass `nullopt` for group here.
  return allocator->alloc(size, device, /*group_name=*/std::nullopt);
}

// Free functor for MemPool
void xpu_symm_free(
    void* ptr,
    size_t /*size*/,
    int /*device*/,
    sycl::queue* /*queue*/) {
  static auto allocator = get_allocator(c10::DeviceType::XPU);
  allocator->free(ptr);
}

// Register allocator for XPU MemPool
struct RegisterXPUMemPoolAllocator {
  RegisterXPUMemPoolAllocator() {
    std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator> allocator =
        torch::xpu::XPUPluggableAllocator::createCustomAllocator(
            xpu_symm_alloc, xpu_symm_free);
    register_mempool_allocator(c10::DeviceType::XPU, allocator);
  }
};

static RegisterXPUMemPoolAllocator register_xpu_mempool_allocator_;

} // namespace
