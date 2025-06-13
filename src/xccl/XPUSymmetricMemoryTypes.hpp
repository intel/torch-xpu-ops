#pragma once

#include <level_zero/ze_api.h>

namespace c10d::symmetric_memory {

constexpr size_t signal_pad_size = 2048;
using HandleType = void*;

} // namespace c10d::symmetric_memory
