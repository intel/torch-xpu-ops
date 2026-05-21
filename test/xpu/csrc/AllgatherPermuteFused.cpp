#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

// ---------------------------------------------------------------------------
// Signal primitives (same semantics as Signal.hpp, inlined for standalone build)
// ---------------------------------------------------------------------------
inline void store_release_u32(uint32_t* addr, uint32_t val) {
  *addr = val;
  sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
}

inline uint32_t load_acquire_u32(uint32_t* addr) {
  sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
  return *addr;
}

// ---------------------------------------------------------------------------
// Fused allgather+permute kernel with in-kernel barrier.
//
// Single kernel launch replaces 4 separate launches:
//   Phase 0: Copy input_shard → symm_mem local slot (all WGs, partitioned)
//   Phase 1: Grid-wide barrier + cross-device signal exchange
//   Phase 2: Allgather + permute (ring-ordered, persistent loop)
//   Phase 3: Post-read grid barrier + cross-device signal exchange
//
// grid_state layout: [pre_counter, pre_flag, post_counter, post_flag]
// Uses generation counter to avoid flag reset between calls.
//
// sync_bufs_ptr[r] = pointer to rank r's sync buffer (symmetric memory).
// Layout per rank: [world_size uint32 for pre-barrier, world_size for post].
// ---------------------------------------------------------------------------

template <typename scalar_t, int VEC_SIZE>
struct AllgatherPermuteFusedVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_ptr;      // local input shard
  const int64_t* rank_ptrs;       // [ws] symm_mem buffer pointers (ALL symm_mem)
  const int32_t* scatter_idx_ptr; // [num_tokens * topk]
  scalar_t* remap_ptr;            // output
  const int64_t* sync_bufs_ptr;   // [ws] sync buffer pointers (symmetric memory)
  int32_t* grid_state;            // [4] device-local atomic state
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;            // hidden_size / VEC_SIZE
  int32_t local_vecs;             // tokens_per_rank * hidden_vecs
  int32_t total_work;             // ws * tokens_per_rank * hidden_vecs
  int32_t generation;

  void operator()(sycl::nd_item<1> item) const {
    const auto tid = static_cast<int32_t>(item.get_local_id(0));
    const auto bid = static_cast<int32_t>(item.get_group(0));
    const auto block_dim = static_cast<int32_t>(item.get_local_range(0));
    const auto grid_dim = static_cast<int32_t>(item.get_group_range(0));
    const int32_t global_id = block_dim * bid + tid;
    const int32_t stride = block_dim * grid_dim;

    // Generation-based flag values (no reset needed between calls)
    const int32_t pre_done_val = 2 * generation + 1;
    const int32_t post_done_val = 2 * generation + 2;

    // ===== Phase 0: Copy input → symm_mem local slot (vectorized) =====
    scalar_t* local_symm = reinterpret_cast<scalar_t*>(rank_ptrs[rank]);
    auto src_v = reinterpret_cast<const vec_t*>(input_ptr);
    auto dst_v = reinterpret_cast<vec_t*>(local_symm);
    for (int32_t i = global_id; i < local_vecs; i += stride) {
      dst_v[i] = src_v[i];
    }

    // ===== Phase 1: Grid barrier + cross-device sync =====
    item.barrier(sycl::access::fence_space::global_space);

    if (tid == 0) {
      sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
          counter(grid_state[0]);
      int32_t old = counter.fetch_add(1);
      if (old + 1 == grid_dim) {
        // Last WG: do cross-device signal exchange
        counter.store(0, sycl::memory_order::relaxed);

        // put_signal to all peers (write 1 to remote sync buffer)
        for (int32_t r = 0; r < world_size; ++r) {
          if (r == rank) continue;
          uint32_t* target = reinterpret_cast<uint32_t*>(sync_bufs_ptr[r]);
          store_release_u32(&target[rank], 1);
        }
        // wait_signal from all peers (spin on local sync buffer)
        uint32_t* my_sync = reinterpret_cast<uint32_t*>(sync_bufs_ptr[rank]);
        for (int32_t r = 0; r < world_size; ++r) {
          if (r == rank) continue;
          while (load_acquire_u32(&my_sync[r]) != 1) {}
          store_release_u32(&my_sync[r], 0);
        }

        // Notify all local WGs
        sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                          sycl::memory_scope::device,
                          sycl::access::address_space::global_space>
            flag(grid_state[1]);
        flag.store(pre_done_val, sycl::memory_order::release);
      }
    }

    // All WGs wait for pre-barrier completion
    if (tid == 0) {
      sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
          flag(grid_state[1]);
      while (flag.load(sycl::memory_order::acquire) != pre_done_val) {}
    }
    item.barrier(sycl::access::fence_space::global_space);

    // ===== Phase 2: Allgather + Permute (ring-ordered, persistent loop) =====
    for (int32_t idx = global_id; idx < total_work; idx += stride) {
      const int32_t vec_h = idx % hidden_vecs;
      const int32_t pair_idx = idx / hidden_vecs;
      const int32_t step = pair_idx % world_size;
      const int32_t local_token_idx = pair_idx / world_size;

      const int32_t src_rank = (rank + step + 1) % world_size;
      const int32_t global_token_idx =
          src_rank * num_tokens_per_rank + local_token_idx;

      // Coalesced vectorized read from source rank
      const scalar_t* src = reinterpret_cast<const scalar_t*>(rank_ptrs[src_rank]);
      auto s_vec = reinterpret_cast<const vec_t*>(
          src + static_cast<int64_t>(local_token_idx) * hidden_size);
      vec_t v = s_vec[vec_h];

      // Write to ALL topk destinations
      const int64_t topk_base = static_cast<int64_t>(global_token_idx) * topk;
      for (int32_t k = 0; k < topk; ++k) {
        const int32_t dst_row = scatter_idx_ptr[topk_base + k];
        auto d_vec = reinterpret_cast<vec_t*>(
            remap_ptr + static_cast<int64_t>(dst_row) * hidden_size);
        d_vec[vec_h] = v;
      }
    }

    // ===== Phase 3: Post-read grid barrier + cross-device sync =====
    item.barrier(sycl::access::fence_space::global_space);

    if (tid == 0) {
      sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
          counter(grid_state[2]);
      int32_t old = counter.fetch_add(1);
      if (old + 1 == grid_dim) {
        counter.store(0, sycl::memory_order::relaxed);

        // Cross-device post-barrier put_signal
        for (int32_t r = 0; r < world_size; ++r) {
          if (r == rank) continue;
          uint32_t* target = reinterpret_cast<uint32_t*>(sync_bufs_ptr[r]);
          store_release_u32(&target[world_size + rank], 1);
        }
        // Cross-device post-barrier wait_signal
        uint32_t* my_sync = reinterpret_cast<uint32_t*>(sync_bufs_ptr[rank]);
        for (int32_t r = 0; r < world_size; ++r) {
          if (r == rank) continue;
          while (load_acquire_u32(&my_sync[world_size + r]) != 1) {}
          store_release_u32(&my_sync[world_size + r], 0);
        }

        // Notify all local WGs
        sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                          sycl::memory_scope::device,
                          sycl::access::address_space::global_space>
            flag(grid_state[3]);
        flag.store(post_done_val, sycl::memory_order::release);
      }
    }

    // All WGs wait for post-barrier completion
    if (tid == 0) {
      sycl::atomic_ref<int32_t, sycl::memory_order::acq_rel,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
          flag(grid_state[3]);
      while (flag.load(sycl::memory_order::acquire) != post_done_val) {}
    }
    item.barrier(sycl::access::fence_space::global_space);
  }
};

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

at::Tensor allgather_permute_fused(
    const at::Tensor& input_shard,
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& scatter_idx,
    at::Tensor remap_hidden_states,
    const at::Tensor& sync_bufs_ptr,
    at::Tensor grid_state,
    int64_t rank,
    int64_t world_size,
    int64_t generation) {
  TORCH_CHECK(
      input_shard.dim() == 2,
      "allgather_permute_fused: input_shard must be 2D");
  TORCH_CHECK(
      input_shard.is_contiguous(),
      "allgather_permute_fused: input_shard must be contiguous");
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "allgather_permute_fused: rank_buffers_ptr must be [world_size]");
  TORCH_CHECK(
      scatter_idx.dim() == 2 && scatter_idx.scalar_type() == at::kInt,
      "allgather_permute_fused: scatter_idx must be 2D int32");
  TORCH_CHECK(
      remap_hidden_states.dim() == 2 && remap_hidden_states.is_contiguous(),
      "allgather_permute_fused: remap must be 2D contiguous");
  TORCH_CHECK(
      sync_bufs_ptr.dim() == 1 && sync_bufs_ptr.size(0) == world_size,
      "allgather_permute_fused: sync_bufs_ptr must be [world_size]");
  TORCH_CHECK(
      grid_state.dim() == 1 && grid_state.size(0) >= 4 &&
          grid_state.scalar_type() == at::kInt,
      "allgather_permute_fused: grid_state must be [>=4] int32");

  const int64_t num_tokens_per_rank = input_shard.size(0);
  const int64_t hidden_size = input_shard.size(1);
  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);

  TORCH_CHECK(num_tokens == num_tokens_per_rank * world_size);
  TORCH_CHECK(remap_hidden_states.size(0) == num_tokens * topk);
  TORCH_CHECK(remap_hidden_states.size(1) == hidden_size);

  if (num_tokens == 0 || topk == 0 || hidden_size == 0) {
    return remap_hidden_states;
  }

  constexpr int VEC_SIZE = 8;

  c10::Device device(c10::DeviceType::XPU, input_shard.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  // Determine number of persistent workgroups.
  // CRITICAL: For grid-wide spin-barrier, ALL WGs must be schedulable
  // simultaneously. Each WG of 512 threads occupies 32 SIMD16 HW threads.
  // On Xe2-HPG, each VE can sustain ~16 HW threads. With max_cu VEs,
  // the safe WG limit = max_cu * 16 / 32 = max_cu / 2.
  // Empirically: 64 WGs works on B60 (160 CUs), 128 deadlocks.
  auto max_cu = queue.get_device().get_info<sycl::info::device::max_compute_units>();
  constexpr int64_t wg_size = 512;
  const int64_t max_wgs = std::max<int64_t>(1, static_cast<int64_t>(max_cu) / 2);

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      input_shard.scalar_type(), "allgather_permute_fused", [&]() {
        TORCH_CHECK(
            hidden_size % VEC_SIZE == 0,
            "allgather_permute_fused: hidden_size must be divisible by VEC_SIZE");
        const int32_t hidden_vecs = static_cast<int32_t>(hidden_size / VEC_SIZE);
        const int32_t local_vecs =
            static_cast<int32_t>(num_tokens_per_rank) * hidden_vecs;
        const int32_t total_work =
            static_cast<int32_t>(world_size * num_tokens_per_rank) * hidden_vecs;

        // Limit WG count: enough work per thread, but ≤ max_cu
        const int64_t min_items_per_thread = 4;
        int64_t needed_wgs =
            (total_work + wg_size * min_items_per_thread - 1) /
            (wg_size * min_items_per_thread);
        int64_t num_wgs = std::min(needed_wgs, max_wgs);
        num_wgs = std::max(num_wgs, int64_t(1));

        auto kfn = AllgatherPermuteFusedVecKernel<scalar_t, VEC_SIZE>{
            input_shard.data_ptr<scalar_t>(),
            rank_buffers_ptr.data_ptr<int64_t>(),
            scatter_idx.data_ptr<int32_t>(),
            remap_hidden_states.data_ptr<scalar_t>(),
            sync_bufs_ptr.data_ptr<int64_t>(),
            grid_state.data_ptr<int32_t>(),
            static_cast<int32_t>(num_tokens_per_rank),
            static_cast<int32_t>(hidden_size),
            static_cast<int32_t>(topk),
            static_cast<int32_t>(rank),
            static_cast<int32_t>(world_size),
            hidden_vecs,
            local_vecs,
            total_work,
            static_cast<int32_t>(generation)};
        sycl_kernel_submit(
            sycl::range<1>(num_wgs * wg_size),
            sycl::range<1>(wg_size),
            queue,
            kfn);
      });

  return remap_hidden_states;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "allgather_permute_fused(Tensor input_shard, "
      "Tensor rank_buffers_ptr, Tensor scatter_idx, "
      "Tensor(a!) remap_hidden_states, Tensor sync_bufs_ptr, "
      "Tensor(b!) grid_state, "
      "int rank, int world_size, int generation) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("allgather_permute_fused", allgather_permute_fused);
}
