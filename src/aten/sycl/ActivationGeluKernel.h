#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void gelu_kernel(TensorIteratorBase& iter, c10::string_view approximate);

void gelu_backward_kernel(
    TensorIteratorBase& iter,
    c10::string_view approximate);

} // namespace xpu
} // namespace native
} // namespace at
