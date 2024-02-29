#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void conj_kernel(TensorIterator& iter);

void neg_conj_kernel(TensorIterator& iter);

void neg_kernel(TensorIterator& iter);

} // at::native::xpu
