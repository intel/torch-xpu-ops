#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/RangeFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/RangeFactoriesKernel.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>

namespace at {

namespace native {
Tensor& arange_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  return xpu::arange_kernel(start, end, step, out);
}
} // namespace native

} // namespace at
