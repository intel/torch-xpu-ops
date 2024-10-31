#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void weight_to_int4pack_kernel(
    const Tensor& weight_packed,
    const Tensor& weight,
    int N,
    int K,
    int fold_len);

} // namespace at::native::xpu