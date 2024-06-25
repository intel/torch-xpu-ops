#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

void amp_non_finite_check_and_unscale_kernel(
    Tensor& scaled_grad,
    Tensor& found_inf,
    const Tensor& inv_scale);

void amp_foreach_non_finite_check_and_unscale_kernel(
    std::vector<std::vector<at::Tensor>> scaled_grads,
    Tensor& found_inf,
    const Tensor& inv_scale);

Tensor& amp_update_scale_kernel(
    Tensor& current_scale,
    Tensor& growth_tracker,
    const Tensor& found_inf,
    double growth_factor,
    double backoff_factor,
    int64_t growth_interval);

} // namespace at::native::xpu
