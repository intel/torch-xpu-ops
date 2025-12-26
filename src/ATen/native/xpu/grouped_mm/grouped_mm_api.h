#pragma once

#include <ATen/core/Tensor.h>

namespace sycltla {

// XPU fast-path implementation for the common MoE case:
//   mat_a: [sum_M, K] (2D)
//   mat_b: [groups, K, N] (3D)
//   offs:  [groups] int32, cumulative end indices over dim0 of mat_a/out
//   out:   [sum_M, N] (2D)
//
// Preconditions (enforced by caller):
// - XPU tensors
// - dtype bf16 for mat_a/mat_b/out
// - contiguous row-major
// - bias is not supported
void grouped_mm_moe_forward(
    at::Tensor mat_a, // bf16
    at::Tensor mat_b, // bf16
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias, // BF16
    at::Tensor &out);

} // namespace sycltla
