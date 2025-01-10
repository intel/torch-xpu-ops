
#pragma once
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void dequant_int4_kernel(
    const Tensor& _weight_int4pack_mm_cuda,
    int qGroupSize,
    const Tensor& weight_scale_zero_point,
    Tensor& weight_dequant);

} // namespace at::native::xpu
