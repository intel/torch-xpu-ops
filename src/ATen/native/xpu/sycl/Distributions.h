#pragma once

#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

void launch_binomial_xpu_kernel(
    TensorIteratorBase& iter,
    XPUGeneratorImpl* gen);

} // namespace at::native::xpu
