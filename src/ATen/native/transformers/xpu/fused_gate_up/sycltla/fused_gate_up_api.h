// Copyright 2020-2026 Intel Corporation
// Licensed under the Apache License, Version 2.0

#pragma once

#include <ATen/ATen.h>

namespace sycltla {

at::Tensor fused_gate_up_silu_sycltla(
    const at::Tensor& input, // [M, K]
    const at::Tensor& gate_weight, // [N, K]
    const at::Tensor& up_weight); // [N, K]

} // namespace sycltla
