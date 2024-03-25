#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void eq_kernel(TensorIteratorBase& iter);

void ne_kernel(TensorIteratorBase& iter);

void lt_kernel(TensorIteratorBase& iter);

void le_kernel(TensorIteratorBase& iter);

void gt_kernel(TensorIteratorBase& iter);

void ge_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
