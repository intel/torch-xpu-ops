/*
 * XPU SymmetricMemory custom Ops
 *
 * Implements `symm_mem::one_shot_all_reduce` and `symm_mem::two_shot_all_reduce_`
 * on XPU, mirroring the CUDA implementation in
 * torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu.
 *
 * Algorithm references:
 *   - Upstream CUDA: CUDASymmetricMemoryOps.cu (one_shot/two_shot kernels)
 *   - BMG SYCL     : sycl-tla/examples/00_bmg_gemm/gemm_allreduce_kernel.hpp
 *                    (allreduce_device<NUM_PER_TH>)
 *
 * Currently supports: dtype ∈ {float32, float16, bfloat16}, op="sum",
 * world_size ∈ {2,4,8}.
 */

#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <comm/SYCLHelpers.h>
#include <torch/library.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <xccl/Signal.hpp>

namespace c10d {
namespace symmetric_memory {

// Anchor symbol referenced by XPUSymmetricMemoryUtils.cpp to prevent
// --as-needed from dropping this SYCL .so as a DT_NEEDED of libtorch_xpu.so.
int xpu_symm_mem_ops_force_link = 0;

// ============================================================================
// Dispatch macros (mirror of CUDA side)
// ============================================================================
#define XPU_INT_SWITCH_CASE(name, val, ...) \
  case val: {                               \
    constexpr int name = val;               \
    __VA_ARGS__();                          \
    break;                                  \
  }

#define XPU_DISPATCH_WORLD_SIZES(world_size, ...)           \
  switch (world_size) {                                     \
    XPU_INT_SWITCH_CASE(k_world_size, 8, __VA_ARGS__);      \
    XPU_INT_SWITCH_CASE(k_world_size, 4, __VA_ARGS__);      \
    XPU_INT_SWITCH_CASE(k_world_size, 2, __VA_ARGS__);      \
    default: {                                              \
      TORCH_CHECK(                                          \
          false,                                            \
          "Not implemented for world_size=",                \
          world_size);                                      \
    }                                                       \
  }

// ============================================================================
// Launch configuration helpers
// ============================================================================
constexpr int kOneShotMaxNumGroups = 24;
constexpr int kOneShotMaxNumThreads = 512;
constexpr int kTwoShotMaxNumGroups = 24;
constexpr int kTwoShotMaxNumThreads = 512;
// 128-bit vectorized load: 16 bytes per thread per iteration.
constexpr int kVecBytes = 16;

template <typename T>
constexpr int elems_per_vec() {
  return kVecBytes / static_cast<int>(sizeof(T));
}

// Opaque 16-byte aligned vector holder so we can do 128-bit loads/stores
// for any scalar type (float / half / bfloat16) without relying on
// sycl::vec<bfloat16,N> arithmetic operators being defined.
template <typename T, int N>
struct alignas(kVecBytes) VecT {
  T data[N];
};

static inline void init_launch_cfg_1d(
    int64_t numel,
    int64_t elems_per_thread,
    int max_groups,
    int max_threads,
    int64_t& num_groups,
    int64_t& num_threads) {
  int64_t total_vec = at::ceil_div(numel, elems_per_thread);
  if (total_vec <= max_threads) {
    num_groups = 1;
    num_threads = std::max<int64_t>(32, (total_vec + 31) / 32 * 32);
  } else {
    num_groups = std::min<int64_t>(
        at::ceil_div(total_vec, (int64_t)max_threads), (int64_t)max_groups);
    num_threads = max_threads;
  }
}

// ============================================================================
// Kernel functors (out of anonymous namespace for SYCL AOT)
// ============================================================================

template <typename scalar_t, int kWorldSize>
struct OneShotAllReduceSumKernel {
  static constexpr int kN = elems_per_vec<scalar_t>();
  using Vec = VecT<scalar_t, kN>;

  scalar_t** peer_ptrs;
  scalar_t* output_ptr;
  int64_t input_offset;
  int64_t numel;
  int my_rank;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t tid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t stride = static_cast<int64_t>(item.get_global_range(0));
    const int64_t vec_total = numel / kN;

    // Rank rotation: each rank starts reading from its own (local) buffer
    // and then fans out to peers in order (my_rank+step) % kWorldSize. This
    // staggers peer reads so at any instant the kWorldSize ranks target
    // distinct peers, avoiding an all-to-one fabric hot-spot on PCIe-P2P.
    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t elem_idx = v * kN + input_offset;
      Vec acc = *reinterpret_cast<const Vec*>(peer_ptrs[my_rank] + elem_idx);
#pragma unroll
      for (int step = 1; step < kWorldSize; ++step) {
        const int p = (my_rank + step) % kWorldSize;
        Vec rhs = *reinterpret_cast<const Vec*>(peer_ptrs[p] + elem_idx);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
          acc.data[i] = static_cast<scalar_t>(
              static_cast<float>(acc.data[i]) +
              static_cast<float>(rhs.data[i]));
        }
      }
      *reinterpret_cast<Vec*>(output_ptr + v * kN) = acc;
    }

    // Tail
    if (tid == 0) {
      for (int64_t i = vec_total * kN; i < numel; ++i) {
        float a = static_cast<float>(peer_ptrs[my_rank][i + input_offset]);
#pragma unroll
        for (int step = 1; step < kWorldSize; ++step) {
          const int p = (my_rank + step) % kWorldSize;
          a += static_cast<float>(peer_ptrs[p][i + input_offset]);
        }
        output_ptr[i] = static_cast<scalar_t>(a);
      }
    }
  }
};

// ============================================================================
// Fused one-shot all-reduce: inlined per-workgroup signal-pad barrier.
// ============================================================================
// Signal-pad layout for this kernel (per rank, in uint32 slots):
//   [kFusedSignalBaseU32 ..                                ) pre-barrier
//   [kFusedSignalBaseU32 + kOneShotMaxNumGroups*kWorldSize ) post-barrier
// Slot [base+region_off + group_id*ws + src_rank] is written by src_rank
// (put_signal into peer's pad) and cleared by the owner rank (wait_signal).
// After an exchange round all slots return to 0, so the region can be reused
// across consecutive fused calls without an explicit reset.
//
// kFusedSignalBaseU32 is placed well beyond the `channel 0` region used by
// symm_mem->barrier(channel=0) so the two paths don't collide if mixed.
constexpr int kFusedSignalBaseU32 = 512;

template <typename scalar_t, int kWorldSize>
struct FusedOneShotAllReduceSumKernel {
  static constexpr int kN = elems_per_vec<scalar_t>();
  using Vec = VecT<scalar_t, kN>;

  scalar_t** peer_ptrs;
  scalar_t* output_ptr;
  uint32_t** signal_pads;
  int64_t input_offset;
  int64_t numel;
  int my_rank;

  static inline uint32_t* slot_of(
      uint32_t** signal_pads,
      int owner_rank,
      int region,
      int group_id,
      int src_rank) {
    const int64_t region_off =
        (int64_t)region * kOneShotMaxNumGroups * kWorldSize;
    return signal_pads[owner_rank] + kFusedSignalBaseU32 + region_off +
        (int64_t)group_id * kWorldSize + src_rank;
  }

  inline void wg_barrier_pre(sycl::nd_item<1> item) const {
    const auto lid = item.get_local_id(0);
    const auto group_id = item.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = static_cast<int>(lid);
      if (peer != my_rank) {
        uint32_t* put_addr = slot_of(
            signal_pads, peer, /*region=*/0, group_id, my_rank);
        uint32_t* wait_addr = slot_of(
            signal_pads, my_rank, /*region=*/0, group_id, peer);
        ::c10d::symmetric_memory::put_signal<
            std::memory_order_release>(put_addr);
        ::c10d::symmetric_memory::wait_signal<
            std::memory_order_acquire>(wait_addr);
      }
    }
    // Gate the non-barrier threads in this WG on the signal exchange above.
    // Local-scope fence is sufficient: the put/wait already issue
    // system-scope atomic_fence(release/acquire) internally, so cross-device
    // memory ordering is already guaranteed.
    item.barrier(sycl::access::fence_space::local_space);
  }

  inline void wg_barrier_post(sycl::nd_item<1> item) const {
    // First ensure all threads in this WG have finished their reads from
    // peer buffers before we signal peers that their buffers are free.
    item.barrier(sycl::access::fence_space::local_space);

    const auto lid = item.get_local_id(0);
    const auto group_id = item.get_group(0);
    if (lid < (size_t)kWorldSize) {
      int peer = static_cast<int>(lid);
      if (peer != my_rank) {
        uint32_t* put_addr = slot_of(
            signal_pads, peer, /*region=*/1, group_id, my_rank);
        uint32_t* wait_addr = slot_of(
            signal_pads, my_rank, /*region=*/1, group_id, peer);
        ::c10d::symmetric_memory::put_signal<
            std::memory_order_release>(put_addr);
        ::c10d::symmetric_memory::wait_signal<
            std::memory_order_acquire>(wait_addr);
      }
    }
    // No trailing item.barrier: nothing in this WG runs after the post
    // barrier; the kernel exits immediately and the XPU stream provides
    // queue-level ordering for the caller.
  }

  void operator()(sycl::nd_item<1> item) const {
    // pre-barrier: all peers have their buffers filled.
    wg_barrier_pre(item);

    const int64_t tid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t stride = static_cast<int64_t>(item.get_global_range(0));
    const int64_t vec_total = numel / kN;

    // Rank rotation: see OneShotAllReduceSumKernel comment.
    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t elem_idx = v * kN + input_offset;
      Vec acc = *reinterpret_cast<const Vec*>(peer_ptrs[my_rank] + elem_idx);
#pragma unroll
      for (int step = 1; step < kWorldSize; ++step) {
        const int p = (my_rank + step) % kWorldSize;
        Vec rhs = *reinterpret_cast<const Vec*>(peer_ptrs[p] + elem_idx);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
          acc.data[i] = static_cast<scalar_t>(
              static_cast<float>(acc.data[i]) +
              static_cast<float>(rhs.data[i]));
        }
      }
      *reinterpret_cast<Vec*>(output_ptr + v * kN) = acc;
    }
    if (tid == 0) {
      for (int64_t i = vec_total * kN; i < numel; ++i) {
        float a = static_cast<float>(peer_ptrs[my_rank][i + input_offset]);
#pragma unroll
        for (int step = 1; step < kWorldSize; ++step) {
          const int p = (my_rank + step) % kWorldSize;
          a += static_cast<float>(peer_ptrs[p][i + input_offset]);
        }
        output_ptr[i] = static_cast<scalar_t>(a);
      }
    }

    // post-barrier: prevent peers from overwriting their buffers before we
    // have finished reading them.
    wg_barrier_post(item);
  }
};

template <typename scalar_t>
struct TwoShotReduceScatterSumKernel {
  static constexpr int kN = elems_per_vec<scalar_t>();
  using Vec = VecT<scalar_t, kN>;

  scalar_t** peer_ptrs;
  scalar_t* self_ptr;
  int64_t shard_start;
  int64_t shard_numel;
  int64_t total_numel;
  int world_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t tid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t stride = static_cast<int64_t>(item.get_global_range(0));
    const int64_t vec_shard = shard_numel / kN;

    for (int64_t v = tid; v < vec_shard; v += stride) {
      const int64_t abs_elem = shard_start + v * kN;
      if (abs_elem + kN > total_numel) break;
      Vec acc = *reinterpret_cast<const Vec*>(peer_ptrs[0] + abs_elem);
      for (int p = 1; p < world_size; ++p) {
        Vec rhs = *reinterpret_cast<const Vec*>(peer_ptrs[p] + abs_elem);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
          acc.data[i] = static_cast<scalar_t>(
              static_cast<float>(acc.data[i]) +
              static_cast<float>(rhs.data[i]));
        }
      }
      *reinterpret_cast<Vec*>(self_ptr + abs_elem) = acc;
    }
  }
};

template <typename scalar_t>
struct TwoShotAllGatherKernel {
  static constexpr int kN = elems_per_vec<scalar_t>();
  using Vec = VecT<scalar_t, kN>;

  scalar_t** peer_ptrs;
  scalar_t* self_ptr;
  int64_t shard_numel;
  int64_t total_numel;
  int world_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t tid = static_cast<int64_t>(item.get_global_linear_id());
    const int64_t stride = static_cast<int64_t>(item.get_global_range(0));
    const int64_t vec_total = total_numel / kN;

    for (int64_t v = tid; v < vec_total; v += stride) {
      const int64_t abs_elem = v * kN;
      int owner = static_cast<int>(abs_elem / shard_numel);
      if (owner >= world_size) owner = world_size - 1;
      Vec val = *reinterpret_cast<const Vec*>(peer_ptrs[owner] + abs_elem);
      *reinterpret_cast<Vec*>(self_ptr + abs_elem) = val;
    }
  }
};

namespace {

// ============================================================================
// Launch helpers
// ============================================================================

template <typename scalar_t, int kWorldSize>
static void launch_one_shot_all_reduce_sum(
    scalar_t** peer_ptrs,
    scalar_t* output_ptr,
    int my_rank,
    int64_t input_offset,
    int64_t numel,
    sycl::queue& q) {
  constexpr int kN = elems_per_vec<scalar_t>();
  int64_t num_groups = 0, num_threads = 0;
  init_launch_cfg_1d(
      numel,
      kN,
      kOneShotMaxNumGroups,
      kOneShotMaxNumThreads,
      num_groups,
      num_threads);

  OneShotAllReduceSumKernel<scalar_t, kWorldSize> ker{
      peer_ptrs, output_ptr, input_offset, numel, my_rank};
  sycl_kernel_submit(
      /*global_range=*/num_groups * num_threads,
      /*local_range=*/num_threads,
      q,
      ker);
}

template <typename scalar_t, int kWorldSize>
static void launch_fused_one_shot_all_reduce_sum(
    scalar_t** peer_ptrs,
    scalar_t* output_ptr,
    uint32_t** signal_pads,
    int my_rank,
    int64_t input_offset,
    int64_t numel,
    sycl::queue& q) {
  constexpr int kN = elems_per_vec<scalar_t>();
  int64_t num_groups = 0, num_threads = 0;
  init_launch_cfg_1d(
      numel,
      kN,
      kOneShotMaxNumGroups,
      kOneShotMaxNumThreads,
      num_groups,
      num_threads);
  // Ensure local_size >= kWorldSize so the first kWorldSize threads can each
  // handle one peer for the inlined barrier.
  if (num_threads < kWorldSize) {
    num_threads = kWorldSize;
  }

  FusedOneShotAllReduceSumKernel<scalar_t, kWorldSize> ker{
      peer_ptrs,
      output_ptr,
      signal_pads,
      input_offset,
      numel,
      my_rank};
  sycl_kernel_submit(
      /*global_range=*/num_groups * num_threads,
      /*local_range=*/num_threads,
      q,
      ker);
}

template <typename scalar_t>
static void launch_two_shot_reduce_scatter_self(
    scalar_t** peer_ptrs,
    scalar_t* self_ptr,
    int64_t shard_start,
    int64_t shard_numel,
    int64_t total_numel,
    int world_size,
    sycl::queue& q) {
  constexpr int kN = elems_per_vec<scalar_t>();
  int64_t num_groups = 0, num_threads = 0;
  init_launch_cfg_1d(
      shard_numel,
      kN,
      kTwoShotMaxNumGroups,
      kTwoShotMaxNumThreads,
      num_groups,
      num_threads);

  TwoShotReduceScatterSumKernel<scalar_t> ker{
      peer_ptrs, self_ptr, shard_start, shard_numel, total_numel, world_size};
  sycl_kernel_submit(
      num_groups * num_threads, num_threads, q, ker);
}

template <typename scalar_t>
static void launch_two_shot_all_gather_self(
    scalar_t** peer_ptrs,
    scalar_t* self_ptr,
    int64_t shard_numel,
    int64_t total_numel,
    int world_size,
    sycl::queue& q) {
  constexpr int kN = elems_per_vec<scalar_t>();
  int64_t num_groups = 0, num_threads = 0;
  init_launch_cfg_1d(
      total_numel,
      kN,
      kTwoShotMaxNumGroups,
      kTwoShotMaxNumThreads,
      num_groups,
      num_threads);

  TwoShotAllGatherKernel<scalar_t> ker{
      peer_ptrs, self_ptr, shard_numel, total_numel, world_size};
  sycl_kernel_submit(
      num_groups * num_threads, num_threads, q, ker);
}

// ============================================================================
// scalar_t dispatch helper
// ============================================================================
// Maps at::ScalarType to the sycl-compatible scalar_t used by the kernels.
// fp32       -> float
// fp16       -> sycl::half
// bfloat16   -> sycl::ext::oneapi::bfloat16
#define XPU_DISPATCH_FLOAT_HALF_BF16(TYPE, NAME, ...)                  \
  [&] {                                                                \
    switch (TYPE) {                                                    \
      case at::kFloat: {                                               \
        using scalar_t = float;                                        \
        return __VA_ARGS__();                                          \
      }                                                                \
      case at::kHalf: {                                                \
        using scalar_t = sycl::half;                                   \
        return __VA_ARGS__();                                          \
      }                                                                \
      case at::kBFloat16: {                                            \
        using scalar_t = sycl::ext::oneapi::bfloat16;                  \
        return __VA_ARGS__();                                          \
      }                                                                \
      default:                                                         \
        TORCH_CHECK(                                                   \
            false,                                                     \
            NAME,                                                      \
            ": unsupported dtype ",                                    \
            TYPE,                                                      \
            " (supported: float32, float16, bfloat16).");              \
    }                                                                  \
  }()

// ============================================================================
// Op entry points
// ============================================================================

at::Tensor one_shot_all_reduce_out_impl(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(
      input.is_contiguous(),
      "one_shot_all_reduce(xpu): input must be contiguous.");
  TORCH_CHECK(
      out.is_contiguous(),
      "one_shot_all_reduce(xpu): output must be contiguous.");
  TORCH_CHECK(
      out.sizes() == input.sizes(),
      "one_shot_all_reduce(xpu): input/output size mismatch.");
  TORCH_CHECK(
      reduce_op == "sum",
      "one_shot_all_reduce(xpu): only sum is supported (got ",
      reduce_op,
      ").");
  const auto dtype = input.scalar_type();
  TORCH_CHECK(
      dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
      "one_shot_all_reduce(xpu): only float32/float16/bfloat16 are supported (got ",
      dtype,
      ").");

  if (input.numel() == 0) {
    return out;
  }

  c10::DeviceGuard guard(input.device());
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "one_shot_all_reduce(xpu): input must be allocated with symm_mem.empty().");
  const int world_size = symm_mem->get_world_size();
  const int my_rank = symm_mem->get_rank();
  const int64_t numel = input.numel();
  const int64_t input_offset = input.storage_offset();

  // Kernel-fused signal-pad barrier (inlined put/wait) is the default path.
  // Set USE_FUSED_ONESHOT=0 to fall back to the two-host-barrier path (uses
  // symm_mem->barrier(channel=0) around a plain reduce kernel).
  static const bool kUseFusedOneShot = []() {
    const char* env = std::getenv("USE_FUSED_ONESHOT");
    return !(env && env[0] == '0');
  }();

  sycl::queue& q = at::xpu::getCurrentXPUStream().queue();

  void** peer_ptrs_raw = symm_mem->get_buffer_ptrs_dev();
  void* out_ptr_raw = out.data_ptr();

  if (kUseFusedOneShot) {
    uint32_t** signal_pads =
        reinterpret_cast<uint32_t**>(symm_mem->get_signal_pad_ptrs_dev());
    XPU_DISPATCH_FLOAT_HALF_BF16(dtype, "one_shot_all_reduce(xpu)", [&]() {
      scalar_t** peer_ptrs = reinterpret_cast<scalar_t**>(peer_ptrs_raw);
      scalar_t* out_ptr = reinterpret_cast<scalar_t*>(out_ptr_raw);
      XPU_DISPATCH_WORLD_SIZES(world_size, [&]() {
        launch_fused_one_shot_all_reduce_sum<scalar_t, k_world_size>(
            peer_ptrs,
            out_ptr,
            signal_pads,
            my_rank,
            input_offset,
            numel,
            q);
      });
    });
    return out;
  }

  symm_mem->barrier(0, /*timeout_ms=*/0);

  XPU_DISPATCH_FLOAT_HALF_BF16(dtype, "one_shot_all_reduce(xpu)", [&]() {
    scalar_t** peer_ptrs = reinterpret_cast<scalar_t**>(peer_ptrs_raw);
    scalar_t* out_ptr = reinterpret_cast<scalar_t*>(out_ptr_raw);
    XPU_DISPATCH_WORLD_SIZES(world_size, [&]() {
      launch_one_shot_all_reduce_sum<scalar_t, k_world_size>(
          peer_ptrs, out_ptr, my_rank, input_offset, numel, q);
    });
  });

  symm_mem->barrier(0, /*timeout_ms=*/0);
  return out;
}

at::Tensor one_shot_all_reduce_xpu(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  auto out = at::empty_like(input);
  return one_shot_all_reduce_out_impl(
      input, std::move(reduce_op), std::move(group_name), out);
}

at::Tensor one_shot_all_reduce_out_xpu(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  return one_shot_all_reduce_out_impl(
      input, std::move(reduce_op), std::move(group_name), out);
}

at::Tensor one_shot_all_reduce_copy_xpu(
    const at::Tensor& symm_buffer,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(
      local_input.is_contiguous() && symm_buffer.is_contiguous(),
      "one_shot_all_reduce_copy(xpu): both tensors must be contiguous.");
  TORCH_CHECK(
      local_input.numel() <= symm_buffer.numel(),
      "one_shot_all_reduce_copy(xpu): local_input must fit in symm_buffer.");
  auto slot = symm_buffer.narrow(0, 0, local_input.numel());
  slot.copy_(local_input);
  auto full = one_shot_all_reduce_xpu(
      symm_buffer, std::move(reduce_op), std::move(group_name));
  return full.narrow(0, 0, local_input.numel()).clone();
}

at::Tensor one_shot_all_reduce_copy_out_xpu(
    const at::Tensor& symm_buffer,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(
      local_input.is_contiguous() && symm_buffer.is_contiguous() &&
          out.is_contiguous(),
      "one_shot_all_reduce_copy_out(xpu): tensors must be contiguous.");
  auto slot = symm_buffer.narrow(0, 0, local_input.numel());
  slot.copy_(local_input);
  auto full_out = at::empty_like(symm_buffer);
  one_shot_all_reduce_out_impl(
      symm_buffer, std::move(reduce_op), std::move(group_name), full_out);
  out.copy_(full_out.narrow(0, 0, local_input.numel()));
  return out;
}

at::Tensor two_shot_all_reduce_out_impl(
    at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor output) {
  TORCH_CHECK(
      input.is_contiguous(),
      "two_shot_all_reduce(xpu): input must be contiguous.");
  TORCH_CHECK(
      output.is_contiguous(),
      "two_shot_all_reduce(xpu): output must be contiguous.");
  TORCH_CHECK(
      output.sizes() == input.sizes(),
      "two_shot_all_reduce(xpu): input/output size mismatch.");
  TORCH_CHECK(
      reduce_op == "sum",
      "two_shot_all_reduce(xpu): only sum is supported.");
  const auto dtype = input.scalar_type();
  TORCH_CHECK(
      dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
      "two_shot_all_reduce(xpu): only float32/float16/bfloat16 are supported (got ",
      dtype,
      ").");

  if (input.numel() == 0) {
    return output;
  }

  c10::DeviceGuard guard(input.device());
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "two_shot_all_reduce(xpu): input must be allocated with symm_mem.empty().");
  const int world_size = symm_mem->get_world_size();
  const int rank = symm_mem->get_rank();
  const int64_t numel = input.numel();

  // Round shard_numel up to the dtype's vec unit so each rank's shard is
  // vec-aligned. Use the worst case (fp32 = 4 elems) which is compatible with
  // bf16/fp16 (8 elems) -> pick the dtype-specific value.
  const int64_t elems_per_vec_val =
      dtype == at::kFloat ? 4 : 8;  // 16B / sizeof(scalar_t)
  int64_t shard_numel =
      ((at::ceil_div(numel, (int64_t)world_size) + elems_per_vec_val - 1) /
       elems_per_vec_val) *
      elems_per_vec_val;
  int64_t shard_start = shard_numel * rank;

  sycl::queue& q = at::xpu::getCurrentXPUStream().queue();

  void** peer_ptrs_raw = symm_mem->get_buffer_ptrs_dev();
  void* self_ptr_raw = symm_mem->get_buffer_ptrs()[rank];

  symm_mem->barrier(0, /*timeout_ms=*/0);
  XPU_DISPATCH_FLOAT_HALF_BF16(dtype, "two_shot_all_reduce(xpu)", [&]() {
    scalar_t** peer_ptrs = reinterpret_cast<scalar_t**>(peer_ptrs_raw);
    scalar_t* self_ptr = reinterpret_cast<scalar_t*>(self_ptr_raw);
    if (shard_start < numel) {
      launch_two_shot_reduce_scatter_self<scalar_t>(
          peer_ptrs,
          self_ptr,
          shard_start,
          shard_numel,
          numel,
          world_size,
          q);
    }
  });

  symm_mem->barrier(0, /*timeout_ms=*/0);
  XPU_DISPATCH_FLOAT_HALF_BF16(dtype, "two_shot_all_reduce(xpu)", [&]() {
    scalar_t** peer_ptrs = reinterpret_cast<scalar_t**>(peer_ptrs_raw);
    scalar_t* self_ptr = reinterpret_cast<scalar_t*>(self_ptr_raw);
    launch_two_shot_all_gather_self<scalar_t>(
        peer_ptrs, self_ptr, shard_numel, numel, world_size, q);
  });
  symm_mem->barrier(0, /*timeout_ms=*/0);

  if (!output.is_same(input)) {
    output.copy_(input);
  }
  return output;
}

at::Tensor two_shot_all_reduce_(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name) {
  two_shot_all_reduce_out_impl(
      input, std::move(reduce_op), std::move(group_name), input);
  return input;
}

at::Tensor two_shot_all_reduce_out(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor output) {
  return two_shot_all_reduce_out_impl(
      input, std::move(reduce_op), std::move(group_name), output);
}

} // namespace

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl(
      "one_shot_all_reduce",
      ::c10d::symmetric_memory::one_shot_all_reduce_xpu);
  m.impl(
      "one_shot_all_reduce_out",
      ::c10d::symmetric_memory::one_shot_all_reduce_out_xpu);
  m.impl(
      "one_shot_all_reduce_copy",
      ::c10d::symmetric_memory::one_shot_all_reduce_copy_xpu);
  m.impl(
      "one_shot_all_reduce_copy_out",
      ::c10d::symmetric_memory::one_shot_all_reduce_copy_out_xpu);
  m.impl(
      "two_shot_all_reduce_",
      ::c10d::symmetric_memory::two_shot_all_reduce_);
  m.impl(
      "two_shot_all_reduce_out",
      ::c10d::symmetric_memory::two_shot_all_reduce_out);
}

} // namespace symmetric_memory
} // namespace c10d
