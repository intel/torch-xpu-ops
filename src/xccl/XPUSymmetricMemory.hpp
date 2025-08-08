#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <xccl/XPUSymmetricMemoryTypes.hpp>

namespace c10d::symmetric_memory {

// Resource wrapper that owns a (vaddr, allocation handle) pair. Upon
// destruction, it unmaps the vaddr and releases the allocation handle.
struct XPUAllocationRef : public c10::intrusive_ptr_target {
  void* ptr;
  HandleType handle;
  size_t block_size;
  int device_idx;

  XPUAllocationRef(
      void* ptr,
      HandleType handle,
      size_t block_size,
      int device_idx);

  ~XPUAllocationRef();
};

class XPUSymmetricMemory : public SymmetricMemory {
 public:
  XPUSymmetricMemory(
      std::vector<c10::intrusive_ptr<XPUAllocationRef>> alloc_refs,
      std::vector<void*> buffers,
      std::vector<void*> signal_pads,
      size_t buffer_size,
      int local_device_idx,
      int rank,
      int world_size);

  ~XPUSymmetricMemory() override = default;

  std::vector<void*> get_buffer_ptrs() override;
  std::vector<void*> get_signal_pad_ptrs() override;
  void** get_buffer_ptrs_dev() override;
  void** get_signal_pad_ptrs_dev() override;
  size_t get_buffer_size() override;
  size_t get_signal_pad_size() override;

  bool has_multicast_support() override;
  void* get_multicast_ptr() override;

  at::Tensor get_buffer(
      int rank,
      c10::IntArrayRef sizes,
      c10::ScalarType dtype,
      int64_t storage_offset) override;

  at::Tensor get_signal_pad(
      int rank,
      c10::IntArrayRef sizes,
      std::optional<c10::ScalarType> dtype,
      int64_t storage_offset) override;

  void barrier(int channel, size_t timeout_ms) override;
  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;
  void copy_buffer(at::Tensor src, at::Tensor dst, size_t size) override;

  int get_rank() override;
  int get_world_size() override;

 private:
  std::vector<c10::intrusive_ptr<XPUAllocationRef>> alloc_refs_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  size_t buffer_size_;
  int local_device_idx_;
  int rank_;
  int world_size_;
  void** buffers_dev_;
  void** signal_pads_dev_;
};

// Metadata associated with each allocation performed by
// `XPUSymmetricMemoryAllocator`.
struct XPUBlock : public c10::intrusive_ptr_target {
  c10::intrusive_ptr<XPUAllocationRef> alloc_ref;
  int device_idx;
  size_t block_size;
  size_t buffer_size;
  size_t signal_pad_offset;
  std::optional<std::string> default_group_name;
  std::map<std::string, c10::intrusive_ptr<XPUSymmetricMemory>> symm_mems;

  XPUBlock(
      c10::intrusive_ptr<XPUAllocationRef> alloc_ref,
      int device_idx,
      size_t block_size,
      size_t buffer_size,
      size_t signal_pad_offset,
      const std::optional<std::string>& group_name);
};

class XPUSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override;

  void free(void* ptr) override;
  size_t get_alloc_size(void* ptr) override;
  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override;
  bool has_multicast_support(int device_idx) override;
  c10::DeviceType supported_device_type() override;
  std::string name() override;

 private:
  c10::intrusive_ptr<XPUBlock> find_block(void* ptr);

  std::shared_mutex mutex_;
  std::unordered_map<void*, c10::intrusive_ptr<XPUBlock>> ptr_to_block_;
};

} // namespace c10d::symmetric_memory
