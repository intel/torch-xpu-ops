#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void gcd_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void lcm_kernel(TensorIteratorBase& iter);

}
} // namespace native
} // namespace at
