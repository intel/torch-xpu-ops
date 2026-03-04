#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {
TORCH_XPU_API void _nested_op_dense_esuhm_xpu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    const NESTED_DENSE_OP& op);
}
