#pragma once

#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

using namespace at::sparse;

// Check if every tensor in a list of tensors matches the current
// device.
bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  Device curDevice = Device(kXPU, c10::xpu::current_device());
  for (const Tensor& t : ts) {
    if (t.device() != curDevice)
      return false;
  }
  return true;
}

TORCH_XPU_API SparseTensor& add_sparse_kernel(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_);

TORCH_XPU_API SparseTensor& mul_sparse_kernel(
    const Tensor& t_,
    const Tensor& src_,
    SparseTensor& r_);

TORCH_XPU_API Tensor _sparse_sum_backward_kernel(
    const Tensor& grad_,
    const SparseTensor& input_,
    IntArrayRef dims_to_sum);

} // namespace at::native::xpu
