/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Splitting the different head dimensions to different files to speed up
// compilation. This file is auto-generated. See "generate_kernels.py"

#include <sycltla/mha_bwd_launch.h>

namespace cute {

template <>
void run_mha_bwd_<cute::half_t, 32, false>(
    sycl::queue& queue,
    FLASH_BWD_params& params) {
  run_mha_bwd_hdim32<cute::half_t, false>(queue, params);
}

template <>
void run_mha_bwd_<cute::half_t, 32, true>(
    sycl::queue& queue,
    FLASH_BWD_params& params) {
  run_mha_bwd_hdim32<cute::half_t, true>(queue, params);
}

template <>
void run_mha_bwd_<cute::bfloat16_t, 32, false>(
    sycl::queue& queue,
    FLASH_BWD_params& params) {
  run_mha_bwd_hdim32<cute::bfloat16_t, false>(queue, params);
}

template <>
void run_mha_bwd_<cute::bfloat16_t, 32, true>(
    sycl::queue& queue,
    FLASH_BWD_params& params) {
  run_mha_bwd_hdim32<cute::bfloat16_t, true>(queue, params);
}

} // namespace cute
