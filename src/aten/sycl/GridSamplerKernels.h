#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace xpu {

Tensor grid_sampler_2d_kernel(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners);

}
} // namespace native
} // namespace at