#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void fill_kernel(TensorIterator& iter, const Scalar& scalar);

}}} // at::native::xpu
