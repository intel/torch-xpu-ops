#pragma once

#include <ATen/ATen.h>
namespace at::native::xpu {

void renorm_kernel(
    const Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm,
    Tensor& out);
}