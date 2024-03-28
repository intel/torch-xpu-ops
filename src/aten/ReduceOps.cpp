#include <ATen/ScalarOps.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <aten/sycl/ScanKernels.h>
#include <aten/sycl/ScanUtils.h>

namespace at {

template <class Stub>
void impl_func_cum_ops(
    const Tensor& self,
    int64_t dim,
    const Tensor& result,
    Stub& stub) {
  NoNamesGuard guard;
  if (self.dim() == 0) {
    result.fill_(self);
  } else if (self.numel() == 0) {
    result.zero_();
  } else {
    dim = maybe_wrap_dim(dim, self.dim());
    stub(result, self.to(result.scalar_type()), dim);
  }
}

Tensor& XPUNativeFunctions::cumsum_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype,
    Tensor& out) {
  impl_func_cum_ops(self, dim, out, at::native::xpu::launch_cumsum_xpu_kernel);
  return out;
}

Tensor XPUNativeFunctions::cumsum(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  ScalarType out_dtype;
  auto is_integral =
      at::isIntegralType(self.scalar_type(), /*includeBool=*/true);
  out_dtype =
      dtype.value_or(is_integral ? ScalarType::Long : self.scalar_type());
  Tensor out = at::empty_strided(
      self.sizes(), self.strides(), self.options().dtype(out_dtype));
  return cumsum_out(self, dim, dtype, out);
}

Tensor& XPUNativeFunctions::cumsum_(
    Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  return cumsum_out(self, dim, dtype, self);
}

} // namespace at