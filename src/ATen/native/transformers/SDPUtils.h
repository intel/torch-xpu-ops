/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/native/transformers/sdp_utils_cpp.h>

namespace sdp {

bool can_use_mem_efficient_attention(sdp::sdp_params params, bool debug);

} // namespace sdp
