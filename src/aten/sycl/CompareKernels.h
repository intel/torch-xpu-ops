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

void clamp_kernel(
    TensorIteratorBase& iter,
    const Scalar& min_value,
    const Scalar& max_value);

void clamp_min_kernel(TensorIteratorBase& iter, const Scalar& min_value);

void clamp_max_kernel(TensorIteratorBase& iter, const Scalar& max_value);

} // namespace xpu
} // namespace native
} // namespace at
