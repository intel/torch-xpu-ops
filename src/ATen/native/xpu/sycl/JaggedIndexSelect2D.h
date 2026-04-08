/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/ATen.h>


namespace at::native::xpu {

  TORCH_XPU_API at::Tensor jagged_index_select_2d_forward_xpu(
    const at::Tensor& values,
    const at::Tensor& indices,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t num_dense_output_rows);

} // namespace at::native::xpu
