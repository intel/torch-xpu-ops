#pragma once

namespace c10d::symmetric_memory {

// Default signal-pad size for each rank's control area.
// 2048 keeps parity with the CUDA-side default and has worked as a practical
// baseline for channelized signaling. This is a default value; higher-level
// symmetric-memory configuration can override the effective pad size.
constexpr size_t signal_pad_size = 2048;
using HandleType = void*;

} // namespace c10d::symmetric_memory
