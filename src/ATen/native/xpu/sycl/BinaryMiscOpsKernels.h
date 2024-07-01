#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

<<<<<<< HEAD:src/ATen/native/xpu/sycl/TensorTransformationsKernel.h
void flip_kernel(TensorIterator& iter, bool quantized);
=======
void mse_kernel(TensorIteratorBase& iter);
>>>>>>> main:src/ATen/native/xpu/sycl/BinaryMiscOpsKernels.h

} // namespace at::native::xpu
