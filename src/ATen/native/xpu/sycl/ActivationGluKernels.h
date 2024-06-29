#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void glu_kernel(TensorIteratorBase& iter);

void glu_backward_kernel(
    const TensorIteratorBase& iter,
    int64_t gI_stride,
    int64_t I_stride);

} // namespace at::native::xpu
