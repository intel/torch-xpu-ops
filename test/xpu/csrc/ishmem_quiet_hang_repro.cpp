#include <ishmem.h>
#include <ishmemx.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int get_env_int(const char* name, int default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  return std::atoi(value);
}

int get_local_rank() {
  constexpr const char* kEnvNames[] = {
      "MPI_LOCALRANKID",
      "OMPI_COMM_WORLD_LOCAL_RANK",
      "PMI_LOCAL_RANK",
      "SLURM_LOCALID",
  };
  for (const char* name : kEnvNames) {
    const char* value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
      return std::atoi(value);
    }
  }
  return 0;
}

std::vector<sycl::device> get_level_zero_gpus() {
  std::vector<sycl::device> devices;
  for (const auto& platform : sycl::platform::get_platforms()) {
    if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero) {
      continue;
    }
    for (const auto& device : platform.get_devices()) {
      if (device.is_gpu()) {
        devices.push_back(device);
      }
    }
  }
  if (devices.empty()) {
    throw std::runtime_error("No Level Zero GPU device found");
  }
  return devices;
}

void log(int rank, const std::string& message) {
  std::cerr << "[ishmem_quiet_repro rank " << rank << "] " << message
            << std::endl;
}

template <typename T>
void check_alloc(T* ptr, const char* name) {
  if (ptr == nullptr) {
    throw std::runtime_error(std::string("allocation failed: ") + name);
  }
}

struct RemoteGetKernel {
  const std::uint8_t* source;
  std::uint8_t* dest;
  std::size_t bytes;
  std::size_t chunk_bytes;
  int source_pe;

  void operator()(sycl::nd_item<1> item) const {
    const std::size_t chunk = item.get_group(0);
    const std::size_t offset = chunk * chunk_bytes;
    std::size_t size = chunk_bytes;
    if (offset + size > bytes) {
      size = bytes - offset;
    }
    ishmemx_getmem_work_group(
        dest + offset, source + offset, size, source_pe, item.get_group());
  }
};

} // namespace

int main() {
  try {
    const int local_rank = get_local_rank();
    const auto devices = get_level_zero_gpus();

    ishmemx_attr_t attr;
    attr.gpu = true;
    attr.initialize_runtime = true;
    attr.device_idx = local_rank % static_cast<int>(devices.size());
    ishmemx_init_attr(&attr);

    const int rank = ishmem_my_pe();
    const int world_size = ishmem_n_pes();
    if (world_size < 2) {
      throw std::runtime_error("Run with at least 2 PEs");
    }

    sycl::queue queue(
        ishmemx_my_device(),
        sycl::async_handler{},
        sycl::property::queue::in_order{});

    const int rma_bytes_env =
        get_env_int("RMA_BYTES", get_env_int("PUT_BYTES", 4 * 1024 * 1024));
    const int chunk_bytes_env =
        get_env_int("RMA_CHUNK_BYTES", get_env_int("PUT_CHUNK_BYTES", 4096));
    const int iterations = get_env_int("QUIET_REPRO_ITERS", 2);
    const int threads = get_env_int("QUIET_REPRO_THREADS", 256);
    if (rma_bytes_env <= 0 || chunk_bytes_env <= 0 || iterations <= 0 ||
        threads <= 0) {
      throw std::runtime_error("Invalid repro size/iteration settings");
    }

    using word_t = std::uint32_t;
    const std::size_t rma_bytes = static_cast<std::size_t>(rma_bytes_env);
    const std::size_t chunk_bytes = static_cast<std::size_t>(chunk_bytes_env);
    const std::size_t num_words =
        (rma_bytes + sizeof(word_t) - 1) / sizeof(word_t);
    const std::size_t alloc_bytes = num_words * sizeof(word_t);
    const std::size_t num_chunks =
        (alloc_bytes + chunk_bytes - 1) / chunk_bytes;
    const int source_pe = (rank + 1) % world_size;

    log(rank, "device_idx=" + std::to_string(attr.device_idx) +
            " device=" +
            queue.get_device().get_info<sycl::info::device::name>() +
            " world_size=" + std::to_string(world_size) +
            " source_pe=" + std::to_string(source_pe) +
            " rma_bytes=" + std::to_string(alloc_bytes) +
            " chunk_bytes=" + std::to_string(chunk_bytes) +
            " num_chunks=" + std::to_string(num_chunks) +
            " iterations=" + std::to_string(iterations));

    word_t* source = static_cast<word_t*>(ishmem_malloc(alloc_bytes));
    word_t* dest =
        sycl::malloc_device<word_t>(num_words, queue.get_device(), queue.get_context());
    word_t* local_input =
        sycl::malloc_device<word_t>(num_words, queue.get_device(), queue.get_context());
    check_alloc(source, "source");
    check_alloc(dest, "dest");
    check_alloc(local_input, "local_input");

    queue.parallel_for(sycl::range<1>(num_words), [=](sycl::id<1> id) {
      const std::size_t i = id[0];
      local_input[i] = static_cast<word_t>((rank + 1) * 1000000 + i);
      source[i] = 0;
      dest[i] = 0;
    });
    queue.wait_and_throw();
    ishmem_barrier_all();

    for (int iter = 0; iter < iterations; ++iter) {
      log(rank, "iter " + std::to_string(iter) + " enqueue memcpy");
      auto copy_event = queue.memcpy(source, local_input, alloc_bytes);
      log(rank, "iter " + std::to_string(iter) + " wait memcpy");
      copy_event.wait_and_throw();
      log(rank, "iter " + std::to_string(iter) + " done memcpy");

      log(rank, "iter " + std::to_string(iter) + " enqueue ishmem barrier");
      auto barrier_event = ishmemx_barrier_all_on_queue(queue, {copy_event});
      log(rank, "iter " + std::to_string(iter) + " wait ishmem barrier");
      barrier_event.wait_and_throw();
      log(rank, "iter " + std::to_string(iter) + " done ishmem barrier");

      log(rank, "iter " + std::to_string(iter) + " submit remote get kernel");
      auto get_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(barrier_event);
        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(num_chunks * static_cast<std::size_t>(threads)),
                sycl::range<1>(static_cast<std::size_t>(threads))),
            RemoteGetKernel{
                reinterpret_cast<const std::uint8_t*>(source),
                reinterpret_cast<std::uint8_t*>(dest),
                alloc_bytes,
                chunk_bytes,
                source_pe});
      });
      log(rank, "iter " + std::to_string(iter) + " wait remote get kernel");
      get_event.wait_and_throw();
      log(rank, "iter " + std::to_string(iter) + " done remote get kernel");
      
      ishmem_quiet();
      /**
      log(rank, "iter " + std::to_string(iter) + " enqueue ishmem quiet");
      auto quiet_event = ishmemx_quiet_on_queue(queue, {get_event});
      log(rank, "iter " + std::to_string(iter) + " wait ishmem quiet");
      quiet_event.wait_and_throw();
      log(rank, "iter " + std::to_string(iter) + " done ishmem quiet");
      **/
    }

    ishmem_barrier_all();
    sycl::free(local_input, queue);
    sycl::free(dest, queue);
    ishmem_free(source);
    ishmem_finalize();

    log(rank, "PASS");
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ishmem_quiet_repro] FAIL: " << e.what() << std::endl;
    return 1;
  }
}
