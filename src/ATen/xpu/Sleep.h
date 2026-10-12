#pragma once
#include <c10/macros/Export.h>
#include <cstdint>

namespace at::xpu {

// Enqueues a kernel that spins for the specified number of cycles
TORCH_XPU_API void sleep(uint64_t cycles);

} // namespace at::xpu
