#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void fill_kernel(TensorIterator& iter, const Scalar& scalar);

}
} // namespace native
} // namespace at
