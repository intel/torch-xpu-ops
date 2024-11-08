#pragma once

#include <ATen/native/transformers/sdp_utils_cpp.h>

namespace sdp {

bool can_use_mem_efficient_attention(sdp::sdp_params params, bool debug);

} // namespace sdp
