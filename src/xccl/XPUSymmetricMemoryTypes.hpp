#pragma once

#include <level_zero/ze_api.h>

namespace c10d::symmetric_memory {

constexpr size_t signal_pad_size = 2048;
// XPU uses Level Zero memory handles for shared memory
using HandleType = ze_ipc_mem_handle_t;

} // namespace c10d::symmetric_memory
