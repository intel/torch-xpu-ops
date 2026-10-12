#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace xpu {

// Single-workgroup top-k kernel (translated from CUDA single-block path).
// One workgroup per slice: radix select to find the k-th value, then gather.
// Good for large dim (≥4096), any batch size. Output is UNSORTED.
bool single_wg_topk_try_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices);

} // namespace xpu
} // namespace native
} // namespace at
