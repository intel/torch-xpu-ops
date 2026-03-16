// Copyright 2020-2026 Intel Corporation
// Licensed under the Apache License, Version 2.0

#include <ATen/core/Tensor.h>
#include <ATen/native/transformers/xpu/fused_gate_up/fused_gate_up_api.h>
#include <torch/library.h>

namespace at::native::xpu {

// XPU dispatch implementation
at::Tensor _fused_gate_up_silu_xpu(
    const at::Tensor& input,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight) {
  return sycltla::fused_gate_up_silu(input, gate_weight, up_weight);
}

} // namespace at::native::xpu