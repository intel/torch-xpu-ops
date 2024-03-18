#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/RangeFactories.h>
#include <torch/library.h>

namespace at {

Tensor& XPUNativeFunctions::arange_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  return at::native::xpu::arange_xpu_out(start, end, step, out);
}

} // namespace at
