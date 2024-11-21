#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void _calculate_moving_average(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    const float averaging_const,
    const int64_t size,
    bool per_row_fake_quant);

TORCH_XPU_API
void _calc_moving_avg_qparams_helper(
    const at::Tensor& x,
    const at::Tensor fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    float* scale_ptr,
    int32_t* zp_ptr,
    int32_t qmin,
    int32_t qmax,
    bool symmetric_quant,
    const int64_t size,
    bool per_row_fq);

} // namespace at::native::xpu
