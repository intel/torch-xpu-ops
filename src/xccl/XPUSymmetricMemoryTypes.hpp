#pragma once

namespace c10d::symmetric_memory {

constexpr size_t signal_pad_size = 2048;
using HandleType = ze_physical_mem_handle_t;

} // namespace c10d::symmetric_memory
