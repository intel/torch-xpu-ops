#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/RenormKernel.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>
namespace at {

void renorm_meta(
    const Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm,
    Tensor& output) {
  TORCH_CHECK(!p.isComplex(), "renorm: p must be real-valued");
  TORCH_CHECK(p.toDouble() > 0.0, "renorm: non-positive-norm not supported");
  TORCH_CHECK(!maxnorm.isComplex(), "renorm: maxnorm must be real-valued");
  TORCH_CHECK(
      maxnorm.toDouble() >= 0.0,
      "renorm: expected maxnorm to be >= 0 but got ",
      maxnorm.toDouble());
  const auto ndim = self.dim();
  TORCH_CHECK(
      ndim > 1,
      "renorm: input needs at least 2 dimensions, got ",
      ndim,
      " dimensions");
  if (output.defined()) {
    xpu::resize_out(output, self.sizes(), {}, self.options());
  } else {
    output = xpu::create_out(self.sizes(), {}, self.options());
  }
}

Tensor& XPUNativeFunctions::renorm_(
    Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm) {
  renorm_meta(self, p, dim, maxnorm, self);
  at::native::xpu::renorm_kernel(self, p, dim, maxnorm, self);
  return self;
}
Tensor& XPUNativeFunctions::renorm_out(
    const Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm,
    Tensor& out) {
  renorm_meta(self, p, dim, maxnorm, out);
  at::native::xpu::renorm_kernel(self, p, dim, maxnorm, out);
  return out;
}
Tensor XPUNativeFunctions::renorm(
    const Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm) {
  Tensor out;
  renorm_meta(self, p, dim, maxnorm, out);
  at::native::xpu::renorm_kernel(self, p, dim, maxnorm, out);
  return out;
}
} // namespace at