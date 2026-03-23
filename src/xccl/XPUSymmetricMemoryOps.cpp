#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <mutex>
#include <unordered_map>

#include <xccl/Signal.hpp>
#include <xccl/XPUSymmetricMemory.hpp>

#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <comm/SYCLContext.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>

namespace c10d::symmetric_memory {

// ============================================================================
// Dispatch macros
// ============================================================================

#define INT_SWITCH_CASE(name, val, ...) \
  case val: {                           \
    constexpr int name = val;           \
    __VA_ARGS__();                      \
    break;                              \
  }

#define DISPATCH_WORLD_SIZES(world_size, ...)      \
  switch (world_size) {                            \
    INT_SWITCH_CASE(k_world_size, 8, __VA_ARGS__); \
    INT_SWITCH_CASE(k_world_size, 4, __VA_ARGS__); \
    INT_SWITCH_CASE(k_world_size, 2, __VA_ARGS__); \
    default: {                                     \
      constexpr int k_world_size = -1;             \
      __VA_ARGS__();                               \
    }                                              \
  }

#define DISPATCH_ALIGNMENTS_16_8_4(alignment, ...)                     \
  switch (alignment) {                                                 \
    INT_SWITCH_CASE(k_alignment, 16, __VA_ARGS__);                     \
    INT_SWITCH_CASE(k_alignment, 8, __VA_ARGS__);                      \
    INT_SWITCH_CASE(k_alignment, 4, __VA_ARGS__);                      \
    default: {                                                         \
      TORCH_CHECK(false, "Not implemented for alignment=", alignment); \
    }                                                                  \
  }

#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))

// ============================================================================
// Helper functions
// ============================================================================

inline size_t get_and_verify_alignment(
    const at::Tensor& input,
    const char* op_name) {
  const size_t min_alignment = std::max(4l, input.element_size());
  const size_t ptr_alignment = at::native::memory::get_alignment(
      static_cast<size_t>(input.storage_offset() * input.element_size()));
  TORCH_CHECK(
      ptr_alignment >= min_alignment,
      op_name,
      "<",
      input.scalar_type(),
      ">: input ptr + offset must be at least ",
      min_alignment,
      "-byte aligned.");

  const size_t size_alignment = at::native::memory::get_alignment(
      static_cast<size_t>(input.numel() * input.element_size()));
  TORCH_CHECK(
      size_alignment >= min_alignment,
      op_name,
      "<",
      input.scalar_type(),
      ">: input size must be at least ",
      min_alignment,
      "-byte aligned.");
  return std::min(ptr_alignment, size_alignment);
}

inline void init_elementwise_launch_config(
    size_t numel,
    size_t element_size,
    size_t alignment,
    size_t splits,
    size_t max_num_blocks,
    size_t max_num_threads,
    int& num_blocks,
    int& num_threads,
    int world_size) {
  const size_t aligned_numel = at::round_up(numel, alignment * splits);
  const size_t numel_per_split = aligned_numel / splits;
  const size_t numel_per_thread = alignment / element_size;

  // XPU sub-group size is typically 16 or 32
  constexpr size_t xpu_subgroup_size = 16;

  if (numel_per_split <= max_num_threads * numel_per_thread) {
    num_blocks = 1;
    num_threads = at::ceil_div(numel_per_split, numel_per_thread);
    num_threads = std::max(num_threads, world_size);
    num_threads = at::round_up(
        static_cast<size_t>(num_threads), xpu_subgroup_size);
  } else {
    num_blocks = std::min(
        at::ceil_div(numel_per_split, max_num_threads * numel_per_thread),
        max_num_blocks);
    num_threads = max_num_threads;
  }
}

// ============================================================================
// Vec type for vectorized memory access (similar to CUDA implementation)
// ============================================================================

template <int Size>
union Vec {
  uint8_t u8[Size];
  uint32_t u32[Size / 4];
  float f32[Size / 4];
};

template <>
union Vec<4> {
  uint8_t u8[4];
  uint32_t u32;
  float f32;
};

// ============================================================================
// Vectorized load/store helpers
// ============================================================================

template <int alignment, typename T>
inline Vec<alignment> ld_vec(const T* ptr) {
  Vec<alignment> vec;
  constexpr int num_elements = alignment / sizeof(T);
  const T* src = ptr;
#pragma unroll
  for (int i = 0; i < num_elements; ++i) {
    reinterpret_cast<T*>(&vec)[i] = src[i];
  }
  return vec;
}

template <int alignment, typename T>
inline void st_vec(T* ptr, const Vec<alignment>& vec) {
  constexpr int num_elements = alignment / sizeof(T);
#pragma unroll
  for (int i = 0; i < num_elements; ++i) {
    ptr[i] = reinterpret_cast<const T*>(&vec)[i];
  }
}

// ============================================================================
// BFloat16 addition helper
// ============================================================================

inline uint32_t add_bf16x2(uint32_t a, uint32_t b) {
  // Extract two bf16 values from each uint32
  uint16_t a_low = static_cast<uint16_t>(a & 0xFFFF);
  uint16_t a_high = static_cast<uint16_t>((a >> 16) & 0xFFFF);
  uint16_t b_low = static_cast<uint16_t>(b & 0xFFFF);
  uint16_t b_high = static_cast<uint16_t>((b >> 16) & 0xFFFF);

  // Convert bf16 to float (bf16 is upper 16 bits of float)
  float fa_low, fa_high, fb_low, fb_high;
  uint32_t tmp;

  tmp = static_cast<uint32_t>(a_low) << 16;
  std::memcpy(&fa_low, &tmp, sizeof(float));
  tmp = static_cast<uint32_t>(a_high) << 16;
  std::memcpy(&fa_high, &tmp, sizeof(float));
  tmp = static_cast<uint32_t>(b_low) << 16;
  std::memcpy(&fb_low, &tmp, sizeof(float));
  tmp = static_cast<uint32_t>(b_high) << 16;
  std::memcpy(&fb_high, &tmp, sizeof(float));

  // Add in float
  float fc_low = fa_low + fb_low;
  float fc_high = fa_high + fb_high;

  // Convert back to bf16
  uint32_t c_low_u32, c_high_u32;
  std::memcpy(&c_low_u32, &fc_low, sizeof(float));
  std::memcpy(&c_high_u32, &fc_high, sizeof(float));

  uint16_t c_low = static_cast<uint16_t>(c_low_u32 >> 16);
  uint16_t c_high = static_cast<uint16_t>(c_high_u32 >> 16);

  return static_cast<uint32_t>(c_low) |
      (static_cast<uint32_t>(c_high) << 16);
}

// ============================================================================
// Vec addition
// ============================================================================

template <int alignment, typename T>
inline Vec<alignment> add_vec(const Vec<alignment>& a, const Vec<alignment>& b) {
  Vec<alignment> c{};
  if constexpr (std::is_same_v<T, float>) {
    if constexpr (alignment == 16) {
      c.f32[0] = a.f32[0] + b.f32[0];
      c.f32[1] = a.f32[1] + b.f32[1];
      c.f32[2] = a.f32[2] + b.f32[2];
      c.f32[3] = a.f32[3] + b.f32[3];
    } else if constexpr (alignment == 8) {
      c.f32[0] = a.f32[0] + b.f32[0];
      c.f32[1] = a.f32[1] + b.f32[1];
    } else if constexpr (alignment == 4) {
      c.f32 = a.f32 + b.f32;
    }
  } else if constexpr (std::is_same_v<T, at::BFloat16>) {
    if constexpr (alignment == 16) {
      c.u32[0] = add_bf16x2(a.u32[0], b.u32[0]);
      c.u32[1] = add_bf16x2(a.u32[1], b.u32[1]);
      c.u32[2] = add_bf16x2(a.u32[2], b.u32[2]);
      c.u32[3] = add_bf16x2(a.u32[3], b.u32[3]);
    } else if constexpr (alignment == 8) {
      c.u32[0] = add_bf16x2(a.u32[0], b.u32[0]);
      c.u32[1] = add_bf16x2(a.u32[1], b.u32[1]);
    } else if constexpr (alignment == 4) {
      c.u32 = add_bf16x2(a.u32, b.u32);
    }
  }
  return c;
}

// ============================================================================
// Load and reduce from all peers
// ============================================================================

// With world_size specialization
template <typename T, int alignment, int k_world_size>
inline std::enable_if_t<(k_world_size > 0), Vec<alignment>> load_and_reduce(
    T** ptrs,
    size_t rank,
    size_t world_size,
    size_t offset) {
  Vec<alignment> vecs[k_world_size];
#pragma unroll
  for (size_t step = 0; step < static_cast<size_t>(k_world_size); ++step) {
    size_t remote_rank = (rank + step) % k_world_size;
    vecs[remote_rank] = ld_vec<alignment>(ptrs[remote_rank] + offset);
  }
  auto acc = vecs[0];
#pragma unroll
  for (size_t r = 1; r < static_cast<size_t>(k_world_size); ++r) {
    acc = add_vec<alignment, T>(acc, vecs[r]);
  }
  return acc;
}

// Without world_size specialization
template <typename T, int alignment, int k_world_size>
inline std::enable_if_t<(k_world_size <= 0), Vec<alignment>> load_and_reduce(
    T** ptrs,
    size_t rank,
    size_t world_size,
    size_t offset) {
  Vec<alignment> acc{};
  for (size_t step = 0; step < world_size; ++step) {
    auto vec = ld_vec<alignment>(ptrs[step] + offset);
    acc = add_vec<alignment, T>(acc, vec);
  }
  return acc;
}

// ============================================================================
// Sync remote blocks (device-side)
// ============================================================================

template <bool hasPrevMemAccess, bool hasSubsequentMemAccess>
inline void sync_remote_blocks_impl(
    uint32_t** signal_pads,
    size_t rank,
    size_t world_size,
    size_t block_idx,
    size_t thread_idx) {
  if (thread_idx < world_size) {
    auto target_rank = thread_idx;
    if constexpr (hasPrevMemAccess) {
      put_signal<std::memory_order_release>(
          signal_pads[target_rank] + block_idx * world_size + rank);
    } else {
      put_signal<std::memory_order_relaxed>(
          signal_pads[target_rank] + block_idx * world_size + rank);
    }
    if constexpr (hasSubsequentMemAccess) {
      wait_signal<std::memory_order_acquire>(
          signal_pads[rank] + block_idx * world_size + target_rank);
    } else {
      wait_signal<std::memory_order_relaxed>(
          signal_pads[rank] + block_idx * world_size + target_rank);
    }
  }
}

// ============================================================================
// Two-shot all-reduce kernel constants
// ============================================================================

constexpr size_t two_shot_all_reduce_max_num_blocks = 24;
constexpr size_t two_shot_all_reduce_max_num_threads = 512;

// ============================================================================
// Two-shot all-reduce kernel (inplace version)
// ============================================================================

template <typename T, int alignment, int k_world_size>
struct TwoShotAllReduceKernelInplace {
  T** input_ptrs;
  size_t input_offset;
  size_t numel;
  uint32_t** signal_pads;
  size_t rank;
  size_t world_size;

  TwoShotAllReduceKernelInplace(
      T** input_ptrs_,
      size_t input_offset_,
      size_t numel_,
      uint32_t** signal_pads_,
      size_t rank_,
      size_t world_size_)
      : input_ptrs(input_ptrs_),
        input_offset(input_offset_),
        numel(numel_),
        signal_pads(signal_pads_),
        rank(rank_),
        world_size(world_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    constexpr size_t numel_per_thread = alignment / sizeof(T);
    auto thread_idx = item.get_local_id(0);
    auto block_idx = item.get_group(0);
    auto block_dim = item.get_local_range(0);
    auto grid_dim = item.get_group_range(0);

    // Pattern 0: Sync before reading from remote buffers
    sync_remote_blocks_impl<false, true>(
        signal_pads, rank, world_size, block_idx, thread_idx);
    item.barrier(sycl::access::fence_space::global_space);

    const size_t numel_per_rank =
        at::round_up(numel, alignment * world_size) / world_size;
    const size_t start = numel_per_rank * rank;

    auto offset = (block_dim * block_idx + thread_idx) * numel_per_thread;
    auto stride = block_dim * grid_dim * numel_per_thread;

    for (size_t i = offset; i < numel_per_rank; i += stride) {
      if (start + i >= numel) {
        continue;
      }
      auto vec = load_and_reduce<T, alignment, k_world_size>(
          input_ptrs, rank, world_size, input_offset + start + i);
      // Store to all remote buffers
      for (size_t step = 0; step < world_size; ++step) {
        size_t remote_rank = (rank + step) % world_size;
        st_vec<alignment>(
            input_ptrs[remote_rank] + input_offset + start + i, vec);
      }
    }

    item.barrier(sycl::access::fence_space::global_space);
    // Pattern 2: Sync after writing
    sync_remote_blocks_impl<true, true>(
        signal_pads, rank, world_size, block_idx, thread_idx);
  }
};

// ============================================================================
// Two-shot all-reduce kernel (with output buffer)
// ============================================================================

template <typename T, int alignment, int k_world_size>
struct TwoShotAllReduceKernel {
  T** input_ptrs;
  T* output_ptr;
  size_t input_offset;
  size_t numel;
  uint32_t** signal_pads;
  size_t rank;
  size_t world_size;

  TwoShotAllReduceKernel(
      T** input_ptrs_,
      T* output_ptr_,
      size_t input_offset_,
      size_t numel_,
      uint32_t** signal_pads_,
      size_t rank_,
      size_t world_size_)
      : input_ptrs(input_ptrs_),
        output_ptr(output_ptr_),
        input_offset(input_offset_),
        numel(numel_),
        signal_pads(signal_pads_),
        rank(rank_),
        world_size(world_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    constexpr size_t numel_per_thread = alignment / sizeof(T);
    auto thread_idx = item.get_local_id(0);
    auto block_idx = item.get_group(0);
    auto block_dim = item.get_local_range(0);
    auto grid_dim = item.get_group_range(0);

    // Pattern 0: Sync before reading from remote buffers
    sync_remote_blocks_impl<false, true>(
        signal_pads, rank, world_size, block_idx, thread_idx);
    item.barrier(sycl::access::fence_space::global_space);

    const size_t numel_per_rank =
        at::round_up(numel, numel_per_thread * world_size) / world_size;
    const size_t start = numel_per_rank * rank;

    auto offset = (block_dim * block_idx + thread_idx) * numel_per_thread;
    auto stride = block_dim * grid_dim * numel_per_thread;

    // Phase 1: Reduce-scatter - each rank reduces its portion
    for (size_t i = offset; i < numel_per_rank; i += stride) {
      if (start + i >= numel) {
        continue;
      }
      auto vec = load_and_reduce<T, alignment, k_world_size>(
          input_ptrs, rank, world_size, input_offset + start + i);
      // Store reduced result to local buffer
      st_vec<alignment>(
          input_ptrs[rank] + input_offset + start + i, vec);
    }

    item.barrier(sycl::access::fence_space::global_space);
    // Sync after reduce-scatter
    sync_remote_blocks_impl<true, true>(
        signal_pads, rank, world_size, block_idx, thread_idx);
    item.barrier(sycl::access::fence_space::global_space);

    // Phase 2: All-gather - copy from all peers to output
    for (size_t i = offset; i < numel_per_rank; i += stride) {
      Vec<alignment> tmp[k_world_size > 0 ? k_world_size : 8];
      size_t actual_world_size =
          k_world_size > 0 ? static_cast<size_t>(k_world_size) : world_size;

      // Load from all peers
      for (size_t step = 0; step < actual_world_size; ++step) {
        size_t remote_rank = (rank + step) % actual_world_size;
        size_t remote_start = numel_per_rank * remote_rank;
        if (remote_start + i < numel) {
          tmp[step] = ld_vec<alignment>(
              input_ptrs[remote_rank] + input_offset + remote_start + i);
        }
      }

      // Store to output
      for (size_t step = 0; step < actual_world_size; ++step) {
        size_t remote_rank = (rank + step) % actual_world_size;
        size_t remote_start = numel_per_rank * remote_rank;
        if (remote_start + i < numel) {
          st_vec<alignment>(output_ptr + remote_start + i, tmp[step]);
        }
      }
    }

    item.barrier(sycl::access::fence_space::global_space);
    // Pattern 2: Sync before exit
    sync_remote_blocks_impl<true, false>(
        signal_pads, rank, world_size, block_idx, thread_idx);
  }
};

// ============================================================================
// Implementation function
// ============================================================================

at::Tensor two_shot_all_reduce_impl(
    at::Tensor input,
    std::optional<at::Tensor> output,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(
      input.is_contiguous(),
      "two_shot_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      reduce_op == "sum",
      "two_shot_all_reduce: only sum is supported for now.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "two_shot_all_reduce: input must be allocated with empty_strided_p2p().");

  const size_t alignment =
      get_and_verify_alignment(input, "two_shot_all_reduce");

  if (output.has_value()) {
    TORCH_CHECK(
        output->is_contiguous(),
        "two_shot_all_reduce: output must be contiguous.");
    const size_t output_alignment =
        get_and_verify_alignment(*output, "two_shot_all_reduce");
    TORCH_CHECK(
        alignment <= output_alignment,
        "two_shot_all_reduce: output alignment must be >= input alignment.");
    TORCH_CHECK(
        output->sizes() == input.sizes(),
        "two_shot_all_reduce: input/output size mismatch.");
    if (input.numel() == 0) {
      TORCH_CHECK(output->scalar_type() == input.scalar_type());
      return *output;
    }
  } else {
    if (input.numel() == 0) {
      return input;
    }
  }

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      symm_mem->get_world_size(),
      two_shot_all_reduce_max_num_blocks,
      two_shot_all_reduce_max_num_threads,
      num_blocks,
      num_threads,
      symm_mem->get_world_size());

  c10::Device device(c10::DeviceType::XPU, input.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  if (!output.has_value()) {
    // Inplace version
    AT_DISPATCH_FLOAT_AND_BFLOAT16(
        input.scalar_type(), "two_shot_all_reduce", [&]() {
          DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
            DISPATCH_WORLD_SIZES(symm_mem->get_world_size(), [&]() {
              using KernelClass = TwoShotAllReduceKernelInplace<
                  scalar_t,
                  k_alignment,
                  k_world_size>;
              auto kfn = KernelClass(
                  reinterpret_cast<scalar_t**>(symm_mem->get_buffer_ptrs_dev()),
                  input.storage_offset(),
                  input.numel(),
                  reinterpret_cast<uint32_t**>(
                      symm_mem->get_signal_pad_ptrs_dev()),
                  symm_mem->get_rank(),
                  symm_mem->get_world_size());
              sycl_kernel_submit(
                  sycl::range<1>(num_blocks * num_threads),
                  sycl::range<1>(num_threads),
                  queue,
                  kfn);
            });
          });
        });
    return input;
  } else {
    // With output buffer
    AT_DISPATCH_FLOAT_AND_BFLOAT16(
        input.scalar_type(), "two_shot_all_reduce", [&]() {
          DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
            DISPATCH_WORLD_SIZES(symm_mem->get_world_size(), [&]() {
              using KernelClass =
                  TwoShotAllReduceKernel<scalar_t, k_alignment, k_world_size>;
              auto kfn = KernelClass(
                  reinterpret_cast<scalar_t**>(symm_mem->get_buffer_ptrs_dev()),
                  output->data_ptr<scalar_t>(),
                  input.storage_offset(),
                  input.numel(),
                  reinterpret_cast<uint32_t**>(
                      symm_mem->get_signal_pad_ptrs_dev()),
                  symm_mem->get_rank(),
                  symm_mem->get_world_size());
              sycl_kernel_submit(
                  sycl::range<1>(num_blocks * num_threads),
                  sycl::range<1>(num_threads),
                  queue,
                  kfn);
            });
          });
        });
    return *output;
  }
}
// ============================================================================
// Low-latency all-reduce using 64-bit data+flag transactions (BF16 only)
// ============================================================================
//
// Each 32-bit data word (2 x bf16) is paired with a 32-bit flag to form
// a 64-bit transaction.  A volatile 64-bit load from remote P2P memory
// guarantees that if the flag is correct the data is too (same word).
//
// Optimizations over the generic path:
//   - BF16 hardcoded: no type dispatch, reduction is always add_bf16x2.
//   - Own-rank fast path: each thread wrote its own index in Phase 1,
//     so it reads back without spinning in Phase 2.
//   - k_world_size template for compile-time unrolling of the rank loop.

constexpr size_t low_latency_all_reduce_max_num_blocks = 24;
constexpr size_t low_latency_all_reduce_max_num_threads = 512;

template <int k_world_size>
struct LowLatencyAllReduceBF16Kernel {
  const uint32_t* scratch_ptr;  // raw bf16 data (2 bf16 per uint32)
  void** buffer_ptrs;           // P2P buffers (one per rank)
  uint32_t* output_ptr;         // output (written as uint32, i.e. 2 bf16)
  size_t input_offset;          // byte offset into each buffer
  size_t num_transactions;      // number of 32-bit data words
  size_t rank;
  size_t world_size;
  uint32_t flag_value;

  LowLatencyAllReduceBF16Kernel(
      const uint32_t* scratch_ptr_,
      void** buffer_ptrs_,
      uint32_t* output_ptr_,
      size_t input_offset_,
      size_t num_transactions_,
      size_t rank_,
      size_t world_size_,
      uint32_t flag_value_)
      : scratch_ptr(scratch_ptr_),
        buffer_ptrs(buffer_ptrs_),
        output_ptr(output_ptr_),
        input_offset(input_offset_),
        num_transactions(num_transactions_),
        rank(rank_),
        world_size(world_size_),
        flag_value(flag_value_) {}

  void operator()(sycl::nd_item<1> item) const {
    const size_t global_id = item.get_global_id(0);
    const size_t global_size = item.get_global_range(0);
    const size_t ws =
        k_world_size > 0 ? static_cast<size_t>(k_world_size) : world_size;

    uint64_t* own_txn_buf = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(buffer_ptrs[rank]) + input_offset);

    for (size_t i = global_id; i < num_transactions; i += global_size) {
      // --- Phase 1: pack own data + flag, write to P2P buffer ----------
      uint32_t own_data = scratch_ptr[i];
      uint64_t own_txn = static_cast<uint64_t>(own_data) |
          (static_cast<uint64_t>(flag_value) << 32);
      own_txn_buf[i] = own_txn;

      // --- Phase 2: read all ranks, reduce -----------------------------
      // Start with own rank directly (no spin needed — we just wrote it).
      uint32_t reduced = own_data;

      // Reduce from all other ranks with spin-read.
#pragma unroll
      for (size_t step = 1; step < ws; ++step) {
        size_t r = (rank + step) % ws;
        volatile uint64_t* remote_ptr = reinterpret_cast<uint64_t*>(
            reinterpret_cast<char*>(buffer_ptrs[r]) + input_offset) + i;

        uint64_t txn;
        do {
          txn = *remote_ptr;
        } while (static_cast<uint32_t>(txn >> 32) != flag_value);

        reduced = add_bf16x2(reduced, static_cast<uint32_t>(txn));
      }

      output_ptr[i] = reduced;
    }
  }
};

at::Tensor low_latency_all_reduce_impl(
    at::Tensor input,
    std::optional<at::Tensor> output,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(
      input.is_contiguous(),
      "low_latency_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      reduce_op == "sum",
      "low_latency_all_reduce: only sum is supported.");
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16,
      "low_latency_all_reduce: only BFloat16 is supported.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "low_latency_all_reduce: input must be allocated with "
      "empty_strided_p2p().");

  const size_t numel = input.numel();
  const size_t data_bytes = numel * sizeof(at::BFloat16);
  const size_t num_transactions =
      at::ceil_div(data_bytes, sizeof(uint32_t));
  const size_t input_offset_bytes =
      input.storage_offset() * sizeof(at::BFloat16);

  TORCH_CHECK(
      symm_mem->get_buffer_size() >=
          input_offset_bytes + num_transactions * sizeof(uint64_t),
      "low_latency_all_reduce: symmetric memory buffer too small. Need ",
      input_offset_bytes + num_transactions * sizeof(uint64_t),
      " bytes, have ",
      symm_mem->get_buffer_size(),
      " bytes.");

  const size_t rank = symm_mem->get_rank();
  const size_t world_size = symm_mem->get_world_size();

  at::Tensor result;
  if (output.has_value()) {
    TORCH_CHECK(
        output->is_contiguous(),
        "low_latency_all_reduce: output must be contiguous.");
    TORCH_CHECK(
        output->sizes() == input.sizes(),
        "low_latency_all_reduce: input/output size mismatch.");
    TORCH_CHECK(
        output->scalar_type() == at::kBFloat16,
        "low_latency_all_reduce: output must be BFloat16.");
    result = *output;
  } else {
    result = input;
  }

  if (numel == 0) {
    return result;
  }

  c10::Device device(c10::DeviceType::XPU, input.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  // Scratch: preserve raw input while P2P buffer is overwritten.
  auto scratch = at::empty_like(input);
  scratch.copy_(input);

  // Per-group monotonic flag counter (all ranks call in same order).
  static std::mutex flag_mutex;
  static std::unordered_map<std::string, uint32_t> flag_counters;
  uint32_t flag_value;
  {
    std::lock_guard<std::mutex> lock(flag_mutex);
    auto& counter = flag_counters[group_name];
    ++counter;
    if (counter == 0)
      ++counter; // skip 0
    flag_value = counter;
  }

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      num_transactions,
      sizeof(uint64_t),
      sizeof(uint64_t),
      1,
      low_latency_all_reduce_max_num_blocks,
      low_latency_all_reduce_max_num_threads,
      num_blocks,
      num_threads,
      world_size);

  DISPATCH_WORLD_SIZES(world_size, [&]() {
    auto kfn = LowLatencyAllReduceBF16Kernel<k_world_size>(
        reinterpret_cast<const uint32_t*>(scratch.data_ptr<at::BFloat16>()),
        symm_mem->get_buffer_ptrs_dev(),
        reinterpret_cast<uint32_t*>(result.data_ptr<at::BFloat16>()),
        input_offset_bytes,
        num_transactions,
        rank,
        static_cast<size_t>(world_size),
        flag_value);
    sycl_kernel_submit(
        sycl::range<1>(num_blocks * num_threads),
        sycl::range<1>(num_threads),
        queue,
        kfn);
  });

  return result;
}

// ============================================================================
// Public API functions
// ============================================================================

at::Tensor all_reduce_low_latency(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name) {
  return low_latency_all_reduce_impl(
      input, std::nullopt, reduce_op, group_name);
}

at::Tensor two_shot_all_reduce_(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name) {
  return two_shot_all_reduce_impl(input, std::nullopt, reduce_op, group_name);
}

at::Tensor two_shot_all_reduce_out(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor output) {
  return two_shot_all_reduce_impl(input, output, reduce_op, group_name);
}

} // namespace c10d::symmetric_memory

// ============================================================================
// Library registration
// ============================================================================

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl(
      "two_shot_all_reduce_",
      c10d::symmetric_memory::two_shot_all_reduce_);
  m.impl(
      "two_shot_all_reduce_out",
      c10d::symmetric_memory::two_shot_all_reduce_out);
  m.impl(
      "all_reduce_low_latency",
      c10d::symmetric_memory::all_reduce_low_latency);
}

