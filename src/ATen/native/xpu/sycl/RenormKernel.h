#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

void renorm_scale_factor_kernel(TensorIteratorBase& iter, double maxnorm);

}
