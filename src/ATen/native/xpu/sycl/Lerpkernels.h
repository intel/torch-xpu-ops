#pragma once

#include <ATen/TensorIterator.h>

namespace at::native::xpu {

void lerp_tensor_kernel(TensorIteratorBase& iter);

void lerp_scalar_kernel(
    TensorIteratorBase& iter,
    const c10::Scalar& weight);

} // namespace at::native::xpu
