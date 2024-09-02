#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void eq_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void ne_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void lt_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void le_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void gt_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void ge_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
