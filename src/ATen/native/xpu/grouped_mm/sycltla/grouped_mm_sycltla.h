#pragma once

#include <ATen/core/Tensor.h>

namespace sycltla {

void grouped_mm_moe_forward_sycltla(
        at::Tensor mat_a, // bf16
        at::Tensor mat_b, // bf16
        std::optional<at::Tensor> offs,
        std::optional<at::Tensor> bias, // BF16
        at::Tensor &out);

} // namespace sycltla
