#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <comm/SYCLHelpers.h>
#include <ishmem.h>
#include <ishmemx.h>
#include <mpi.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

namespace {

struct IshmemAllgatherPermuteState {
  std::mutex mutex;
  bool initialized = false;
  void* symm_input = nullptr;
  size_t symm_input_bytes = 0;
  at::Tensor staging;
};

IshmemAllgatherPermuteState& get_state() {
  static IshmemAllgatherPermuteState state;
  return state;
}

bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool debug_enabled() {
  return env_enabled("ALLGATHER_PERMUTE_ISHMEM_DEBUG");
}

bool sync_debug_enabled() {
  return env_enabled("ALLGATHER_PERMUTE_ISHMEM_SYNC_DEBUG");
}

void debug_log(int64_t rank, const char* message) {
  if (debug_enabled()) {
    std::cerr << "[allgather_permute_ishmem rank " << rank << "] "
              << message << std::endl;
  }
}

void ensure_ishmem_initialized(int device_index) {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.initialized) {
    return;
  }

  int initialized = 0;
  ishmemx_query_initialized(&initialized);
  if (!initialized) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    ishmemx_attr_t attr;
    attr.device_idx = device_index;
    attr.gpu = true;
    attr.initialize_runtime = !mpi_initialized;
    ishmemx_init_attr(&attr);
  }
  state.initialized = true;
}

void ensure_symmetric_input(size_t input_bytes) {
  constexpr size_t kDefaultInputBufferBytes = 8 * 1024 * 1024;
  const size_t allocation_bytes =
      std::max(input_bytes, kDefaultInputBufferBytes);
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.symm_input_bytes >= input_bytes) {
    return;
  }

  if (state.symm_input != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.symm_input);
    ishmem_barrier_all();
    state.symm_input = nullptr;
    state.symm_input_bytes = 0;
  }

  state.symm_input = ishmem_malloc(allocation_bytes);
  TORCH_CHECK(
      state.symm_input != nullptr,
      "allgather_permute_ishmem: ishmem_malloc failed for ",
      allocation_bytes,
      " bytes");
  state.symm_input_bytes = allocation_bytes;

  // Symmetric heap allocation is collective; keep all PEs aligned before use.
  ishmem_barrier_all();
}

void* current_symmetric_input() {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.symm_input;
}

at::Tensor ensure_staging(
    const at::TensorOptions& options,
    int64_t total_tokens,
    int64_t hidden_size) {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  const int64_t required_numel = total_tokens * hidden_size;
  if (!state.staging.defined() || state.staging.numel() < required_numel ||
      state.staging.scalar_type() != options.dtype().toScalarType() ||
      state.staging.device() != options.device()) {
    state.staging = at::empty({total_tokens, hidden_size}, options);
  }
  return state.staging;
}

template <typename scalar_t>
struct AllgatherPermuteIshmemKernel {
  const scalar_t* input_ptr;
  const uint8_t* symm_input_ptr;
  scalar_t* staging_ptr;
  const int32_t* scatter_idx_ptr;
  scalar_t* remap_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t total_token_groups;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t token_group =
        static_cast<int32_t>(item.get_group(0));
    if (token_group >= total_token_groups) {
      return;
    }

    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));
    const int32_t step = token_group / num_tokens_per_rank;
    const int32_t local_token_idx = token_group % num_tokens_per_rank;
    const int32_t src_rank = (rank + step + 1) % world_size;
    const int32_t global_token_idx =
        src_rank * num_tokens_per_rank + local_token_idx;

    scalar_t* staging_row =
        staging_ptr + static_cast<int64_t>(token_group) * hidden_size;
    const int64_t row_bytes =
        static_cast<int64_t>(hidden_size) * sizeof(scalar_t);

    if (src_rank == rank) {
      const scalar_t* src_row =
          input_ptr + static_cast<int64_t>(local_token_idx) * hidden_size;
      for (int32_t h = lid; h < hidden_size; h += lsize) {
        staging_row[h] = src_row[h];
      }
      sycl::group_barrier(item.get_group());
    } else {
      const uint8_t* remote_row =
          symm_input_ptr + static_cast<int64_t>(local_token_idx) * row_bytes;
      ishmemx_getmem_work_group(
          static_cast<void*>(staging_row),
          static_cast<const void*>(remote_row),
          static_cast<size_t>(row_bytes),
          src_rank,
          item.get_group());
    }

    const int32_t scatter_base = global_token_idx * topk;
    for (int32_t h = lid; h < hidden_size; h += lsize) {
      const scalar_t value = staging_row[h];
      for (int32_t k = 0; k < topk; ++k) {
        const int32_t dst_row = scatter_idx_ptr[scatter_base + k];
        remap_ptr[static_cast<int64_t>(dst_row) * hidden_size + h] = value;
      }
    }
  }
};

} // namespace

at::Tensor allgather_permute_ishmem(
    const at::Tensor& input_shard,
    const at::Tensor& scatter_idx,
    at::Tensor remap_hidden_states,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      input_shard.dim() == 2,
      "allgather_permute_ishmem: input_shard must be 2D");
  TORCH_CHECK(
      input_shard.is_contiguous(),
      "allgather_permute_ishmem: input_shard must be contiguous");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "allgather_permute_ishmem: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "allgather_permute_ishmem: scatter_idx must be int32");
  TORCH_CHECK(
      scatter_idx.is_contiguous(),
      "allgather_permute_ishmem: scatter_idx must be contiguous");
  TORCH_CHECK(
      remap_hidden_states.dim() == 2,
      "allgather_permute_ishmem: remap_hidden_states must be 2D");
  TORCH_CHECK(
      remap_hidden_states.is_contiguous(),
      "allgather_permute_ishmem: remap_hidden_states must be contiguous");
  TORCH_CHECK(
      input_shard.scalar_type() == remap_hidden_states.scalar_type(),
      "allgather_permute_ishmem: input_shard and remap_hidden_states dtype must match");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "allgather_permute_ishmem: rank must be in [0, world_size)");

  const int64_t num_tokens_per_rank = input_shard.size(0);
  const int64_t hidden_size = input_shard.size(1);
  const int64_t num_tokens = scatter_idx.size(0);
  const int64_t topk = scatter_idx.size(1);
  TORCH_CHECK(
      num_tokens == num_tokens_per_rank * world_size,
      "allgather_permute_ishmem: scatter_idx first dim must equal input_shard.size(0) * world_size");
  TORCH_CHECK(
      remap_hidden_states.size(0) == num_tokens * topk,
      "allgather_permute_ishmem: remap_hidden_states first dim must be num_tokens * topk");
  TORCH_CHECK(
      remap_hidden_states.size(1) == hidden_size,
      "allgather_permute_ishmem: remap_hidden_states hidden size mismatch");

  if (num_tokens == 0 || topk == 0 || hidden_size == 0) {
    return remap_hidden_states;
  }

  c10::Device device(c10::DeviceType::XPU, input_shard.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  debug_log(rank, "enter op");
  debug_log(rank, "before ensure_ishmem_initialized");
  ensure_ishmem_initialized(input_shard.device().index());
  debug_log(rank, "after ensure_ishmem_initialized");
  TORCH_CHECK(
      ishmem_my_pe() == rank,
      "allgather_permute_ishmem: ISHMEM PE does not match rank");
  TORCH_CHECK(
      ishmem_n_pes() == world_size,
      "allgather_permute_ishmem: ISHMEM PE count does not match world_size");

  const size_t input_bytes =
      static_cast<size_t>(input_shard.numel() * input_shard.element_size());
  debug_log(rank, "before ensure_symmetric_input");
  ensure_symmetric_input(input_bytes);
  debug_log(rank, "after ensure_symmetric_input");
  void* symm_input = current_symmetric_input();
  debug_log(rank, "before ensure_staging");
  at::Tensor staging = ensure_staging(
      input_shard.options(), num_tokens, hidden_size);
  debug_log(rank, "after ensure_staging");

  debug_log(rank, "enqueue memcpy");
  auto copy_event =
      queue.memcpy(symm_input, input_shard.data_ptr(), input_bytes);
  if (sync_debug_enabled()) {
    debug_log(rank, "wait memcpy");
    copy_event.wait_and_throw();
    debug_log(rank, "done memcpy");
  }

  debug_log(rank, "enqueue ishmem barrier");
  auto barrier_event = ishmemx_barrier_all_on_queue(queue, {copy_event});
  if (sync_debug_enabled()) {
    debug_log(rank, "wait ishmem barrier");
    barrier_event.wait_and_throw();
    debug_log(rank, "done ishmem barrier");
  }

  constexpr int64_t threads = 256;
  const int64_t total_token_groups = num_tokens;

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      remap_hidden_states.scalar_type(), "allgather_permute_ishmem", [&]() {
        auto kfn = AllgatherPermuteIshmemKernel<scalar_t>{
            input_shard.data_ptr<scalar_t>(),
            static_cast<const uint8_t*>(symm_input),
            staging.data_ptr<scalar_t>(),
            scatter_idx.data_ptr<int32_t>(),
            remap_hidden_states.data_ptr<scalar_t>(),
            static_cast<int32_t>(num_tokens_per_rank),
            static_cast<int32_t>(hidden_size),
            static_cast<int32_t>(topk),
            static_cast<int32_t>(rank),
            static_cast<int32_t>(world_size),
            static_cast<int32_t>(total_token_groups)};

        debug_log(rank, "submit AllgatherPermuteIshmemKernel");
        auto kernel_event = queue.submit([&](sycl::handler& cgh) {
          cgh.depends_on(barrier_event);
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(total_token_groups * threads),
                  sycl::range<1>(threads)),
              kfn);
        });
        debug_log(rank, "submitted AllgatherPermuteIshmemKernel");

        debug_log(rank, "wait AllgatherPermuteIshmemKernel before ishmem quiet");
        kernel_event.wait_and_throw();
        debug_log(rank, "done AllgatherPermuteIshmemKernel");
        debug_log(rank, "run host ishmem quiet");
        ishmem_quiet();
        debug_log(rank, "done host ishmem quiet");
      });

  debug_log(rank, "return op");
  return remap_hidden_states;
}

void allgather_permute_ishmem_finalize(const at::Tensor&) {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.initialized) {
    ishmem_finalize();
    state.initialized = false;
  }
  state.symm_input = nullptr;
  state.symm_input_bytes = 0;
  state.staging.reset();
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "allgather_permute_ishmem(Tensor input_shard, Tensor scatter_idx, "
      "Tensor(a!) remap_hidden_states, int rank, int world_size) -> Tensor(a!)");
  m.def("allgather_permute_ishmem_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("allgather_permute_ishmem", allgather_permute_ishmem);
  m.impl(
      "allgather_permute_ishmem_finalize",
      allgather_permute_ishmem_finalize);
}
