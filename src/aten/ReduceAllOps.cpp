#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/ReduceMaxValuesKernel.h>
#include <aten/sycl/ReduceMinValuesKernel.h>
#include <torch/library.h>

#include <iostream>

namespace at {

void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = native::make_reduction(
      "max_all", result, input, IntArrayRef{}, false, dtype);
  native::xpu::max_all_launch_kernel(iter);
}

Tensor& XPUNativeFunctions::max_out(const Tensor& self, Tensor& out) {
  // First check if the devices match (CPU vs GPU)
  TORCH_CHECK(self.device() == out.device());

  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()), typeMetaToScalarType(out.dtype())));

  at::native::resize_output(out, {});

  max_all_kernel_impl(out, self.contiguous());
  return out;
}

Tensor XPUNativeFunctions::max(const Tensor& self) {
  TORCH_CHECK(
      self.numel() > 0,
      "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  max_all_kernel_impl(result, self.contiguous());
  return result;
}

void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = native::make_reduction(
      "min_all", result, input, IntArrayRef{}, false, dtype);
  native::xpu::min_all_launch_kernel(iter);
}

Tensor& XPUNativeFunctions::min_out(const Tensor& self, Tensor& out) {
  // First check if the devices match (CPU vs GPU)
  TORCH_CHECK(self.device() == out.device());

  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()), typeMetaToScalarType(out.dtype())));

  at::native::resize_output(out, {});

  min_all_kernel_impl(out, self.contiguous());
  return out;
}

Tensor XPUNativeFunctions::min(const Tensor& self) {
  TORCH_CHECK(
      self.numel() > 0,
      "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  min_all_kernel_impl(result, self.contiguous());
  return result;
}

} // namespace at
