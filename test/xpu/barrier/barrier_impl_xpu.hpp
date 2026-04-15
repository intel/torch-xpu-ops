#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

namespace test_xpu_barrier {

sycl::event barrier_impl_xpu(
    uint32_t** signal_pads,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    sycl::queue& queue);

sycl::event barrier_impl_xpu_all_ranks(
    uint32_t** signal_pads,
    int channel,
    int world_size,
    size_t timeout_ms,
    sycl::queue& queue);

} // namespace test_xpu_barrier
