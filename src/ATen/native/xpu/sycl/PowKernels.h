#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void pow_tensor_scalar_kernel(
    TensorIteratorBase& iter,
    const Scalar& exp_scalar);

void pow_tensor_tensor_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
