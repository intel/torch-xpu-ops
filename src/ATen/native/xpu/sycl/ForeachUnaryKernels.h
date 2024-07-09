#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

std::vector<Tensor> foreach_sqrt_kernel(TensorList tensors);

void foreach_sqrt_kernel_(TensorList tensors);

} // namespace at::native::xpu
