#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void renorm_scale_factor_kernel(
    TensorIteratorBase& iter,
    double maxnorm);

}
