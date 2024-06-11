#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/ReduceMaxValuesKernel.h>
#include <aten/sycl/ReduceMinValuesKernel.h>
#include <torch/library.h>

namespace at {

void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = native::make_reduction(
      "min_all", result, input, IntArrayRef{}, false, dtype);
  native::xpu::min_all_launch_kernel(iter);
}

Tensor XPUNativeFunctions::min(const Tensor& self) {
  TORCH_CHECK(
      self.numel() > 0,
      "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  min_all_kernel_impl(result, self.contiguous());
  return result;
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

void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = native::make_reduction(
      "max_all", result, input, IntArrayRef{}, false, dtype);
  native::xpu::max_all_launch_kernel(iter);
}

Tensor XPUNativeFunctions::max(const Tensor& self) {
  TORCH_CHECK(
      self.numel() > 0,
      "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  max_all_kernel_impl(result, self.contiguous());
  return result;
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

static void check_amax_amin(
    const char* name,
    const Tensor& self,
    IntArrayRef dim,
    const Tensor& out) {
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      name,
      " got illegal dtype for self, and out:",
      self.scalar_type(),
      out.scalar_type());
  if (self.numel() == 0) {
    native::zero_numel_check_dims(self, dim, name);
  }
}

Tensor& XPUNativeFunctions::amax_out(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    Tensor& out) {
  check_amax_amin("amax()", self, dim, out);
  max_all_kernel_impl(out, self.contiguous());
  return out;
}

Tensor XPUNativeFunctions::amax(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  Tensor out = at::empty({0}, self.options());
  check_amax_amin("amax()", self, dim, out);
  max_all_kernel_impl(out, self.contiguous());
  return out;
}

} // namespace at
