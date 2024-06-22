#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void abs_kernel(TensorIteratorBase& iter);

<<<<<<< HEAD
void erf_kernel(TensorIteratorBase& iter);

void erfc_kernel(TensorIteratorBase& iter);

=======
>>>>>>> main
} // namespace at::native::xpu
