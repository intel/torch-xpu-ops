#pragma once

#include <c10/macros/Macros.h>
#include <ATen/ATen.h>

#define XPUSHMEM_CHECK(stmt, msg)                                             \
  do {                                                                       \
    int result = (stmt);                                                     \
    TORCH_CHECK(                                                             \
        result == 0,                                                         \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + msg + \
            ". Error code: " + std::to_string(result));                      \
  } while (0)

namespace c10d::xpushmem_extension {

// Check if NVSHMEM is available
TORCH_API bool is_ishmem_available();

// Initializes the device state in CUmodule so that it’s able to perform NVSHMEM
// operations.
TORCH_API void ishmemx_cumodule_init(uintptr_t module);

TORCH_API void ishmem_put(at::Tensor& tensor, const int64_t peer);

TORCH_API void ishmem_get(at::Tensor& tensor, const int64_t peer);

TORCH_API void ishmem_wait_for_signal(at::Tensor& sigpad, int64_t signal, int64_t peer);

TORCH_API void ishmem_put_with_signal(at::Tensor& tensor, at::Tensor& sigpad, int64_t signal, int64_t peer);

} // namespace c10d::nvshmem_extension
