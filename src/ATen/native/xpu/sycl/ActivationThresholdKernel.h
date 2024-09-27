#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void threshold_kernel(
    TensorIteratorBase& iter,
    const Scalar& threshold,
    const Scalar& value);

}
} // namespace native
} // namespace at
