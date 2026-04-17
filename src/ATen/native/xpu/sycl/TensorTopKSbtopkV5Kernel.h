#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API bool sbtopk_v5_try_launch(
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
