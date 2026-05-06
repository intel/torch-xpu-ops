/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/transformers/xpu/fused_gate_up/fused_gate_up_api.h>
#include <ATen/native/transformers/xpu/fused_gate_up/sycltla/fused_gate_up_api.h>

namespace sycltla {

bool is_fused_gate_up_silu_available() {
#ifndef USE_SYCLTLA
  return false;
#else
  return true;
#endif
}

at::Tensor fused_gate_up_silu(
    const at::Tensor& input,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false,
      "fused_gate_up_silu: Torch XPU was not compiled with SYCLTLA support.");
  return at::Tensor();
#else
  return fused_gate_up_silu_sycltla(input, gate_weight, up_weight);
#endif
}

} // namespace sycltla