/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/Tensor.h>
#include <ATen/native/transformers/xpu/fused_gate_up/fused_gate_up_api.h>
#include <torch/library.h>

namespace at::native::xpu {

at::Tensor _fused_gate_up_silu_xpu(
    const at::Tensor& input,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight) {
  return sycltla::fused_gate_up_silu(input, gate_weight, up_weight);
}

} // namespace at::native::xpu

TORCH_LIBRARY_FRAGMENT(xpu, m) {
  m.def(
      "_fused_gate_up_silu(Tensor input, Tensor gate_weight, "
      "Tensor up_weight) -> Tensor");
  m.def("_is_fused_gate_up_silu_available() -> bool");
}

TORCH_LIBRARY_IMPL(xpu, XPU, m) {
  m.impl("_fused_gate_up_silu", at::native::xpu::_fused_gate_up_silu_xpu);
}

// No Tensor args → dispatcher can't infer XPU key; use CatchAll
TORCH_LIBRARY_IMPL(xpu, CatchAll, m) {
  m.impl("_is_fused_gate_up_silu_available", []() {
    return sycltla::is_fused_gate_up_silu_available();
  });
}
