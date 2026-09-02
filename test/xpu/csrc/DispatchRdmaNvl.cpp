// Four/eight-rank token dispatch. The lower and upper rank halves are two
// switch domains; traffic inside a half uses direct GPU IPC and traffic
// crossing the halves is packed and sent by RDMA.
// The op submits exactly one SYCL kernel. After the inter-rank barrier, plain
// queue memcpy commands expose the symmetric receive regions to PyTorch.

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <ishmem.h>
#include <ishmemx.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace {

constexpr int32_t kMaxWorldSize = 8;
constexpr int32_t kDefaultThreads = 512;
constexpr int32_t kDefaultChannels = 16;

struct int4_t {
  int x, y, z, w;
};

template <typename T>
inline void st_na_global_v(T* ptr, T value) {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(sizeof(T) == 16, "st_na_global_v requires 16-byte values");
  using vec4_t = uint32_t __attribute__((ext_vector_type(4)));
  vec4_t tmp;
  __builtin_memcpy(&tmp, &value, 16);
  auto* addr = reinterpret_cast<void*>(ptr);
  asm volatile(
      "lsc_store.ugm.wb.wb (M1, 32) flat[%0]:a64 %1:d32x4"
      :
      : "rw"(addr), "rw"(tmp)
      : "memory");
#else
  *ptr = value;
#endif
}

template <typename T>
inline T ld_nc_global_v(const T* ptr) {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(sizeof(T) == 16, "ld_nc_global_v requires 16-byte values");
  using vec4_t = uint32_t __attribute__((ext_vector_type(4)));
  vec4_t tmp;
  auto* addr = reinterpret_cast<const void*>(ptr);
  asm volatile(
      "lsc_load.ugm.uc.ca (M1, 32) %0:d32x4 flat[%1]:a64"
      : "=rw"(tmp)
      : "rw"(addr));
  T result;
  __builtin_memcpy(&result, &tmp, 16);
  return result;
#else
  return *ptr;
#endif
}

inline void copy_bytes_wg(
    uint8_t* dst,
    const uint8_t* src,
    size_t bytes,
    sycl::nd_item<1> item) {
  const size_t lid = item.get_local_id(0);
  const size_t lsize = item.get_local_range(0);
  if (((reinterpret_cast<uintptr_t>(dst) |
        reinterpret_cast<uintptr_t>(src)) &
       15) != 0) {
    for (size_t i = lid; i < bytes; i += lsize) {
      dst[i] = src[i];
    }
    return;
  }
  const size_t vectors = bytes / sizeof(int4_t);
  auto* dst4 = reinterpret_cast<int4_t*>(dst);
  auto* src4 = reinterpret_cast<const int4_t*>(src);
  for (size_t i = lid; i < vectors; i += lsize) {
    st_na_global_v(dst4 + i, ld_nc_global_v(src4 + i));
  }
  const size_t tail = bytes % sizeof(int4_t);
  if (lid < tail) {
    const size_t base = vectors * sizeof(int4_t);
    dst[base + lid] = src[base + lid];
  }
}

inline void copy_bytes_multi_wg(
    const std::array<uint8_t*, kMaxWorldSize>& destinations,
    uint32_t destination_mask,
    size_t destination_offset,
    const uint8_t* src,
    size_t bytes,
    sycl::nd_item<1> item) {
  const size_t lid = item.get_local_id(0);
  const size_t lsize = item.get_local_range(0);
  uintptr_t alignment_bits = reinterpret_cast<uintptr_t>(src);
  for (int32_t dst = 0; dst < kMaxWorldSize; ++dst) {
    if (destination_mask & (1u << dst)) {
      alignment_bits |=
          reinterpret_cast<uintptr_t>(
              destinations[dst] + destination_offset);
    }
  }
  if ((alignment_bits & 15) != 0) {
    for (size_t i = lid; i < bytes; i += lsize) {
      const uint8_t value = src[i];
      for (int32_t dst = 0; dst < kMaxWorldSize; ++dst) {
        if (destination_mask & (1u << dst)) {
          destinations[dst][destination_offset + i] = value;
        }
      }
    }
    return;
  }

  const size_t vectors = bytes / sizeof(int4_t);
  auto* src4 = reinterpret_cast<const int4_t*>(src);
  for (size_t i = lid; i < vectors; i += lsize) {
    const int4_t value = ld_nc_global_v(src4 + i);
    for (int32_t dst = 0; dst < kMaxWorldSize; ++dst) {
      if (destination_mask & (1u << dst)) {
        auto* dst4 =
            reinterpret_cast<int4_t*>(
                destinations[dst] + destination_offset);
        st_na_global_v(dst4 + i, value);
      }
    }
  }
  const size_t tail = bytes % sizeof(int4_t);
  if (lid < tail) {
    const size_t base = vectors * sizeof(int4_t);
    const uint8_t value = src[base + lid];
    for (int32_t dst = 0; dst < kMaxWorldSize; ++dst) {
      if (destination_mask & (1u << dst)) {
        destinations[dst][destination_offset + base + lid] = value;
      }
    }
  }
}

template <typename T>
constexpr T align_up(T value, T alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct PayloadLayout {
  size_t hidden_off;
  size_t src_rank_off;
  size_t topk_idx_off;
  size_t topk_weights_off;
  size_t bytes_per_token;

  PayloadLayout(
      size_t hidden_bytes,
      size_t topk_idx_bytes,
      size_t topk_weights_bytes)
      : hidden_off(0),
        src_rank_off(align_up<size_t>(hidden_bytes, alignof(int32_t))),
        topk_idx_off(
            align_up<size_t>(src_rank_off + sizeof(int32_t), 16)),
        topk_weights_off(topk_idx_off + topk_idx_bytes),
        bytes_per_token(align_up<size_t>(
            topk_weights_off + topk_weights_bytes, 16)) {}
};

struct SymmetricLayout {
  size_t count_off;
  size_t send_payload_off;
  size_t recv_payload_off;
  size_t total_bytes;

  SymmetricLayout(
      int32_t num_ranks,
      int32_t ranks_per_switch,
      int32_t num_channels,
      int64_t cap,
      size_t bytes_per_token) {
    const size_t send_slots =
        static_cast<size_t>(ranks_per_switch) * cap;
    const size_t recv_slots = static_cast<size_t>(num_ranks) * cap;
    count_off = 0;
    send_payload_off = align_up<size_t>(
        static_cast<size_t>(num_ranks) * num_channels *
            sizeof(uint64_t),
        128);
    recv_payload_off = align_up<size_t>(
        send_payload_off + send_slots * bytes_per_token, 128);
    total_bytes =
        align_up<size_t>(
            recv_payload_off + recv_slots * bytes_per_token, 128);
  }
};

struct State {
  std::mutex mutex;
  bool initialized = false;
  void* symmetric = nullptr;
  size_t symmetric_bytes = 0;
};

State& state() {
  static State value;
  return value;
}

int positive_env(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value != nullptr && value[0] != '\0') {
    const int parsed = std::atoi(value);
    if (parsed > 0) {
      return parsed;
    }
  }
  return fallback;
}

void ensure_ishmem_initialized(int device_index) {
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.initialized) {
    return;
  }
  int initialized = 0;
  ishmemx_query_initialized(&initialized);
  if (!initialized) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    ishmemx_attr_t attr{};
    attr.device_idx = device_index;
    attr.gpu = true;
    attr.initialize_runtime = !mpi_initialized;
    ishmemx_init_attr(&attr);
  }
  s.initialized = true;
}

uint8_t* ensure_symmetric(size_t bytes) {
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.symmetric != nullptr && s.symmetric_bytes >= bytes) {
    return static_cast<uint8_t*>(s.symmetric);
  }
  if (s.symmetric != nullptr) {
    ishmem_barrier_all();
    ishmem_free(s.symmetric);
    ishmem_barrier_all();
  }
  const size_t alloc = std::max<size_t>(bytes, 1 * 1024 * 1024);
  s.symmetric = ishmem_malloc(alloc);
  TORCH_CHECK(
      s.symmetric != nullptr,
      "dispatch_rdma_nvl: ishmem_malloc failed for ",
      alloc,
      " bytes");
  s.symmetric_bytes = alloc;
  ishmem_barrier_all();
  return static_cast<uint8_t*>(s.symmetric);
}

struct DispatchRdmaNvlKernel {
  const uint8_t* x;
  const int64_t* topk_idx;
  const float* topk_weights;
  const bool* is_token_in_rank;

  uint8_t* send_payload;
  uint8_t* recv_payload;
  uint64_t* recv_count;

  std::array<uint8_t*, kMaxWorldSize> ipc_recv_payloads;
  std::array<uint64_t*, kMaxWorldSize> ipc_recv_counts;

  int32_t rank;
  int32_t num_ranks;
  int32_t ranks_per_switch;
  int32_t num_tokens;
  int32_t num_channels;
  int32_t cap;
  int32_t chunk_tokens;
  size_t hidden_bytes;
  size_t topk_idx_bytes;
  size_t topk_weights_bytes;
  PayloadLayout payload_layout;

  void operator()(sycl::nd_item<1> item) const {
    const auto group = item.get_group();
    const auto subgroup = item.get_sub_group();
    const int32_t channel =
        static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lane_id =
        static_cast<int32_t>(subgroup.get_local_id()[0]);

    const int32_t token_begin =
        static_cast<int32_t>(
            static_cast<int64_t>(num_tokens) * channel / num_channels);
    const int32_t token_end =
        static_cast<int32_t>(
            static_cast<int64_t>(num_tokens) * (channel + 1) /
            num_channels);
    const int32_t slot_begin = token_begin;
    const int32_t slot_end = token_end;
    const int32_t channel_capacity = slot_end - slot_begin;
    int32_t counts[kMaxWorldSize] = {};
    int32_t sent_counts[kMaxWorldSize] = {};
    for (int32_t token = token_begin; token < token_end; ++token) {
      const bool* route =
          is_token_in_rank + static_cast<size_t>(token) * num_ranks;
      uint32_t destination_mask = 0;
      if (lane_id == 0) {
        for (int32_t dst = 0; dst < num_ranks; ++dst) {
          if (route[dst]) {
            destination_mask |= (1u << dst);
          }
        }
      }
      destination_mask =
          sycl::group_broadcast(subgroup, destination_mask, 0);
      if (destination_mask == 0) {
        continue;
      }

      std::array<uint8_t*, kMaxWorldSize> payloads{};
      for (int32_t dst = 0; dst < num_ranks; ++dst) {
        if ((destination_mask & (1u << dst)) == 0) {
          continue;
        }
        const int32_t count = counts[dst];
        if (count >= channel_capacity) {
          destination_mask &= ~(1u << dst);
          continue;
        }
        const bool is_ipc =
            dst / ranks_per_switch == rank / ranks_per_switch;
        const int32_t remote_lane = dst % ranks_per_switch;
        if (is_ipc) {
          uint8_t* recv_base = ipc_recv_payloads[dst];
          const size_t output_slot =
              static_cast<size_t>(rank) * cap + slot_begin + count;
          payloads[dst] =
              recv_base + output_slot * payload_layout.bytes_per_token;
        } else {
          const size_t output_slot =
              static_cast<size_t>(remote_lane) * cap +
              slot_begin + count;
          payloads[dst] =
              send_payload +
              output_slot * payload_layout.bytes_per_token;
        }
      }
      if (destination_mask == 0) {
        continue;
      }

      copy_bytes_multi_wg(
          payloads,
          destination_mask,
          payload_layout.hidden_off,
          x + static_cast<size_t>(token) * hidden_bytes,
          hidden_bytes,
          item);
      copy_bytes_multi_wg(
          payloads,
          destination_mask,
          payload_layout.topk_idx_off,
          reinterpret_cast<const uint8_t*>(topk_idx) +
              static_cast<size_t>(token) * topk_idx_bytes,
          topk_idx_bytes,
          item);
      copy_bytes_multi_wg(
          payloads,
          destination_mask,
          payload_layout.topk_weights_off,
          reinterpret_cast<const uint8_t*>(topk_weights) +
              static_cast<size_t>(token) * topk_weights_bytes,
          topk_weights_bytes,
          item);
      if (lid == 0) {
        for (int32_t dst = 0; dst < num_ranks; ++dst) {
          if (destination_mask & (1u << dst)) {
            *reinterpret_cast<int32_t*>(
                payloads[dst] + payload_layout.src_rank_off) = rank;
          }
        }
      }

      uint32_t ready_remote_mask = 0;
      for (int32_t dst = 0; dst < num_ranks; ++dst) {
        if ((destination_mask & (1u << dst)) == 0) {
          continue;
        }
        const int32_t new_count = ++counts[dst];
        const bool is_ipc =
            dst / ranks_per_switch == rank / ranks_per_switch;
        if (!is_ipc &&
            new_count - sent_counts[dst] >= chunk_tokens) {
          ready_remote_mask |= (1u << dst);
        }
      }
      if (ready_remote_mask != 0) {
        item.barrier(
            sycl::access::fence_space::global_and_local);
        sycl::atomic_fence(
            sycl::memory_order::release,
            sycl::memory_scope::device);
      }
      for (int32_t dst = 0; dst < num_ranks; ++dst) {
        if ((ready_remote_mask & (1u << dst)) == 0) {
          continue;
        }
        const int32_t remote_lane = dst % ranks_per_switch;
        const int32_t new_count = counts[dst];
        const size_t staging_slot =
            static_cast<size_t>(remote_lane) * cap + slot_begin +
            sent_counts[dst];
        const size_t remote_slot =
            static_cast<size_t>(rank) * cap + slot_begin +
            sent_counts[dst];
        const unsigned int qp =
            static_cast<unsigned int>(
                channel * ranks_per_switch + remote_lane);
        const int32_t ready = new_count - sent_counts[dst];
        ishmemx_putmem_nbi_work_group_qp(
            recv_payload +
                remote_slot * payload_layout.bytes_per_token,
            send_payload +
                staging_slot * payload_layout.bytes_per_token,
            static_cast<size_t>(ready) *
                payload_layout.bytes_per_token,
            dst,
            group,
            qp);
        sent_counts[dst] = new_count;
      }
    }
    item.barrier(sycl::access::fence_space::global_and_local);

    const size_t count_index =
        static_cast<size_t>(rank) * num_channels + channel;
    sycl::atomic_fence(
        sycl::memory_order::release, sycl::memory_scope::system);
    for (int32_t dst = 0; dst < num_ranks; ++dst) {
      const int32_t count = counts[dst];
      const bool is_ipc =
          dst / ranks_per_switch == rank / ranks_per_switch;
      if (is_ipc) {
        if (lid == 0) {
          uint64_t* count_base = ipc_recv_counts[dst];
          sycl::atomic_ref<
              uint64_t,
              sycl::memory_order::relaxed,
              sycl::memory_scope::system,
              sycl::access::address_space::global_space>
              count_ref(count_base[count_index]);
          count_ref.store(
              static_cast<uint64_t>(count),
              sycl::memory_order::release);
        }
        item.barrier(sycl::access::fence_space::local_space);
        continue;
      }

      const int32_t remote_lane = dst % ranks_per_switch;
      const size_t staging_slot =
          static_cast<size_t>(remote_lane) * cap + slot_begin +
          sent_counts[dst];
      const size_t remote_slot =
          static_cast<size_t>(rank) * cap + slot_begin +
          sent_counts[dst];
      const unsigned int qp =
          static_cast<unsigned int>(
          channel * ranks_per_switch + remote_lane);
      const int32_t tail_count = count - sent_counts[dst];
      if (tail_count > 0) {
        ishmemx_putmem_nbi_work_group_qp(
            recv_payload +
                remote_slot * payload_layout.bytes_per_token,
            send_payload +
                staging_slot * payload_layout.bytes_per_token,
            static_cast<size_t>(tail_count) *
                payload_layout.bytes_per_token,
            dst,
            group,
            qp);
      }
      ishmemx_fence_work_group_qp(dst, group, qp);
      if (lid == 0) {
        ishmemx_uint64_atomic_set_nbi_qp(
            recv_count + count_index,
            static_cast<uint64_t>(count),
            dst,
            qp);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    ishmemx_quiet_work_group(group);
  }
};

void check_collective_layout(
    const at::Tensor& x,
    int64_t hidden,
    int64_t topk,
    int64_t cap,
    int64_t num_channels_arg) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  TORCH_CHECK(
      mpi_initialized,
      "dispatch_rdma_nvl requires an MPI-initialized launch");
  int64_t local[6] = {
      x.size(0),
      hidden,
      topk,
      static_cast<int64_t>(x.element_size()),
      cap,
      num_channels_arg};
  int64_t minimum[6];
  int64_t maximum[6];
  MPI_Allreduce(local, minimum, 6, MPI_INT64_T, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(local, maximum, 6, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  for (int i = 0; i < 6; ++i) {
    TORCH_CHECK(
        minimum[i] == maximum[i],
        "dispatch_rdma_nvl requires identical layouts on all ranks; field ",
        i,
        " differs (min=",
        minimum[i],
        ", max=",
        maximum[i],
        ")");
  }
}

} // namespace

at::Tensor dispatch_rdma_nvl(
    const at::Tensor& x,
    const at::Tensor& topk_idx,
    const at::Tensor& topk_weights,
    const at::Tensor& is_token_in_rank,
    at::Tensor recv_payload,
    at::Tensor recv_channel_counts,
    int64_t rank,
    int64_t num_ranks,
    int64_t num_max_tokens_per_rank,
    int64_t num_channels_arg) {
  TORCH_CHECK(
      num_ranks == 4 || num_ranks == 8,
      "dispatch_rdma_nvl requires 4 or 8 ranks");
  TORCH_CHECK(
      rank >= 0 && rank < num_ranks,
      "dispatch_rdma_nvl: rank must be in [0, num_ranks)");
  const int32_t ranks_per_switch =
      static_cast<int32_t>(num_ranks / 2);
  TORCH_CHECK(
      x.device().is_xpu() && x.dim() == 2 && x.is_contiguous(),
      "dispatch_rdma_nvl: x must be a contiguous 2D XPU tensor");
  TORCH_CHECK(
      topk_idx.scalar_type() == at::kLong && topk_idx.dim() == 2 &&
          topk_idx.is_contiguous() && topk_idx.device() == x.device(),
      "dispatch_rdma_nvl: topk_idx must be contiguous int64 on x.device()");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat &&
          topk_weights.sizes() == topk_idx.sizes() &&
          topk_weights.is_contiguous() &&
          topk_weights.device() == x.device(),
      "dispatch_rdma_nvl: topk_weights must be contiguous float32 on "
      "x.device() and match topk_idx");
  TORCH_CHECK(
      is_token_in_rank.scalar_type() == at::kBool &&
          is_token_in_rank.sizes() ==
              at::IntArrayRef({x.size(0), num_ranks}) &&
          is_token_in_rank.is_contiguous() &&
          is_token_in_rank.device() == x.device(),
      "dispatch_rdma_nvl: is_token_in_rank must be contiguous bool "
      "[num_tokens, num_ranks] on x.device()");
  TORCH_CHECK(
      topk_idx.size(0) == x.size(0),
      "dispatch_rdma_nvl: x and topk tensors must have the same num_tokens");

  const int64_t cap = num_max_tokens_per_rank;
  const int64_t hidden = x.size(1);
  const int64_t topk = topk_idx.size(1);
  TORCH_CHECK(cap > 0, "dispatch_rdma_nvl: capacity must be positive");
  TORCH_CHECK(
      cap >= x.size(0),
      "dispatch_rdma_nvl: num_max_tokens_per_rank must be >= num_tokens "
      "so channel-local packing cannot drop tokens");

  int32_t num_channels =
      num_channels_arg > 0
      ? static_cast<int32_t>(num_channels_arg)
      : positive_env("DISPATCH_RDMA_NVL_CHANNELS", kDefaultChannels);
  num_channels = std::max<int32_t>(num_channels, 1);

  c10::Device device(c10::DeviceType::XPU, x.device().index());
  c10::DeviceGuard guard(device);
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  ensure_ishmem_initialized(x.device().index());
  TORCH_CHECK(
      ishmem_my_pe() == rank && ishmem_n_pes() == num_ranks,
      "dispatch_rdma_nvl: ISHMEM PE topology must match num_ranks");
  check_collective_layout(x, hidden, topk, cap, num_channels_arg);

  const int qps_per_pe =
      std::max(positive_env("ISHMEM_IBGDA_QPS_PER_PE", 1), 1);
  TORCH_CHECK(
      qps_per_pe >= num_channels * ranks_per_switch,
      "dispatch_rdma_nvl: ISHMEM_IBGDA_QPS_PER_PE must be at least ",
      num_channels * ranks_per_switch);
  const int32_t chunk_tokens = std::max(
      positive_env("DISPATCH_RDMA_NVL_CHUNK_TOKENS", 8), 1);
  const int32_t threads =
      positive_env("DISPATCH_RDMA_NVL_THREADS", kDefaultThreads);
  TORCH_CHECK(
      threads % 32 == 0,
      "dispatch_rdma_nvl: DISPATCH_RDMA_NVL_THREADS must be a multiple of 32");
  TORCH_CHECK(
      threads <= static_cast<int32_t>(
          queue.get_device()
              .get_info<sycl::info::device::max_work_group_size>()),
      "dispatch_rdma_nvl: DISPATCH_RDMA_NVL_THREADS exceeds the device "
      "max_work_group_size");

  const size_t hidden_bytes =
      static_cast<size_t>(hidden) * x.element_size();
  const size_t topk_idx_bytes =
      static_cast<size_t>(topk) * sizeof(int64_t);
  const size_t topk_weights_bytes =
      static_cast<size_t>(topk) * sizeof(float);
  const PayloadLayout payload_layout(
      hidden_bytes, topk_idx_bytes, topk_weights_bytes);
  TORCH_CHECK(
      recv_payload.scalar_type() == at::kByte &&
          recv_payload.sizes() == at::IntArrayRef(
              {num_ranks,
               cap,
               static_cast<int64_t>(payload_layout.bytes_per_token)}) &&
          recv_payload.is_contiguous() &&
          recv_payload.device() == x.device(),
      "dispatch_rdma_nvl: recv_payload must be contiguous uint8 "
      "[num_ranks, cap, bytes_per_token]");
  TORCH_CHECK(
      recv_channel_counts.scalar_type() == at::kLong &&
          recv_channel_counts.sizes() ==
              at::IntArrayRef({num_ranks, num_channels}) &&
          recv_channel_counts.is_contiguous() &&
          recv_channel_counts.device() == x.device(),
      "dispatch_rdma_nvl: recv_channel_counts must be int64 "
      "[num_ranks, num_channels]");
  const SymmetricLayout layout(
      static_cast<int32_t>(num_ranks),
      ranks_per_switch,
      num_channels,
      cap,
      payload_layout.bytes_per_token);
  uint8_t* symmetric = ensure_symmetric(layout.total_bytes);
  auto* symmetric_count =
      reinterpret_cast<uint64_t*>(symmetric + layout.count_off);
  uint8_t* send_payload = symmetric + layout.send_payload_off;
  uint8_t* symmetric_recv_payload =
      symmetric + layout.recv_payload_off;

  const size_t count_slots =
      static_cast<size_t>(num_ranks) * num_channels;
  queue.memset(
      symmetric_count,
      0,
      layout.send_payload_off - layout.count_off);
  queue.wait_and_throw();
  ishmem_barrier_all();

  std::array<uint8_t*, kMaxWorldSize> ipc_recv_payloads{};
  std::array<uint64_t*, kMaxWorldSize> ipc_recv_counts{};
  const int32_t switch_begin =
      static_cast<int32_t>(rank / ranks_per_switch) *
      ranks_per_switch;
  const int32_t switch_end = switch_begin + ranks_per_switch;
  for (int32_t dst = switch_begin; dst < switch_end; ++dst) {
    if (dst == rank) {
      ipc_recv_payloads[dst] = symmetric_recv_payload;
      ipc_recv_counts[dst] = symmetric_count;
      continue;
    }
    ipc_recv_payloads[dst] = static_cast<uint8_t*>(
        ishmem_ptr(symmetric_recv_payload, dst));
    ipc_recv_counts[dst] = static_cast<uint64_t*>(
        ishmem_ptr(symmetric_count, dst));
    TORCH_CHECK(
        ipc_recv_payloads[dst] != nullptr &&
            ipc_recv_counts[dst] != nullptr,
        "dispatch_rdma_nvl: rank ",
        rank,
        " cannot obtain IPC pointers for same-switch peer ",
        dst);
  }

  const int64_t work_groups = num_channels;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(
                static_cast<size_t>(work_groups) * threads),
            sycl::range<1>(threads)),
        DispatchRdmaNvlKernel{
            static_cast<const uint8_t*>(x.data_ptr()),
            topk_idx.data_ptr<int64_t>(),
            topk_weights.data_ptr<float>(),
            is_token_in_rank.data_ptr<bool>(),
            send_payload,
            symmetric_recv_payload,
            symmetric_count,
            ipc_recv_payloads,
            ipc_recv_counts,
            static_cast<int32_t>(rank),
            static_cast<int32_t>(num_ranks),
            ranks_per_switch,
            static_cast<int32_t>(x.size(0)),
            num_channels,
            static_cast<int32_t>(cap),
            chunk_tokens,
            hidden_bytes,
            topk_idx_bytes,
            topk_weights_bytes,
            payload_layout});
  });
  queue.wait_and_throw();
  ishmem_barrier_all();

  queue.memcpy(
      recv_payload.data_ptr(),
      symmetric_recv_payload,
      static_cast<size_t>(num_ranks) * cap *
          payload_layout.bytes_per_token);
  queue.memcpy(
      recv_channel_counts.data_ptr(),
      symmetric_count,
      count_slots * sizeof(uint64_t));
  queue.wait_and_throw();
  return recv_payload;
}

void dispatch_rdma_nvl_finalize(const at::Tensor&) {
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.symmetric != nullptr) {
    ishmem_barrier_all();
    ishmem_free(s.symmetric);
    s.symmetric = nullptr;
    s.symmetric_bytes = 0;
  }
  if (s.initialized) {
    int initialized = 0;
    ishmemx_query_initialized(&initialized);
    if (initialized) {
      ishmem_finalize();
    }
    s.initialized = false;
  }
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "dispatch_rdma_nvl(Tensor x, Tensor topk_idx, Tensor topk_weights, "
      "Tensor is_token_in_rank, Tensor(a!) recv_payload, "
      "Tensor(b!) recv_channel_counts, int rank, int num_ranks, "
      "int num_max_tokens_per_rank, int num_channels) -> Tensor(a!)");
  m.def("dispatch_rdma_nvl_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("dispatch_rdma_nvl", dispatch_rdma_nvl);
  m.impl("dispatch_rdma_nvl_finalize", dispatch_rdma_nvl_finalize);
}
