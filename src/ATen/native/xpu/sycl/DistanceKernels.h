#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {
void cdist_kernel(
    Tensor& result,
    const Tensor& x1_expanded,
    const Tensor& x2_expanded,
    double p);

Tensor cdist_backward_kernel(
    Tensor& grad_x1,
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const double p,
    const Tensor& cdist);

void pdist_forward_kernel(Tensor& result, const Tensor& self, double p);

void pdist_backward_kernel(
    Tensor& result,
    const Tensor& grad,
    const Tensor& self,
    const double p,
    const Tensor& dist);

} // namespace at::native::xpu
