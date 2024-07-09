#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/RangeFactoriesKernel.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <torch/library.h>

namespace at {

Tensor& XPUNativeFunctions::arange_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  return at::native::xpu::arange_kernel(start, end, step, out);
}

Tensor& XPUNativeFunctions::range_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  return at::native::xpu::range_kernel(start, end, step, out);
}

} // namespace at
