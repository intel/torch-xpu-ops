#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
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

Tensor& renorm_impl(
    const Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm,
    Tensor& out) {
  auto self_sizes = self.sizes();
  dim = c10::maybe_wrap_dim(dim, self_sizes.size());

  DimVector reduce_dims(self_sizes.size());
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  auto dtype = self.scalar_type();
  auto acc_type = at::toAccumulateType(dtype, c10::DeviceType::XPU);
  Tensor norm;
  if (acc_type != dtype) {
    norm = at::linalg_vector_norm(
        self,
        p.toDouble(),
        reduce_dims,
        /*keepdim=*/true,
        /*dtype=*/acc_type);
  } else {
    norm = at::linalg_vector_norm(
        self,
        p.toDouble(),
        reduce_dims,
        /*keepdim=*/true);
  }

  auto factor = (acc_type == c10::toRealValueType(dtype))
      ? norm
      : at::empty(norm.sizes(), self.options());
  auto iter = TensorIteratorConfig()
                  .add_output(factor)
                  .add_input(norm)
                  .set_check_mem_overlap(false)
                  .cast_common_dtype_to_outputs(true)
                  .build();

  at::native::xpu::renorm_scale_factor_kernel(iter, maxnorm.toDouble());
  return at::mul_outf(self, factor, const_cast<Tensor&>(out));
}

Tensor& XPUNativeFunctions::renorm_(
    Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm) {
  renorm_meta(self, p, dim, maxnorm, self);
  renorm_impl(self, p, dim, maxnorm, self);
  return self;
}
Tensor& XPUNativeFunctions::renorm_out(
    const Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm,
    Tensor& out) {
  renorm_meta(self, p, dim, maxnorm, out);
  renorm_impl(self, p, dim, maxnorm, out);
  return out;
}
Tensor XPUNativeFunctions::renorm(
    const Tensor& self,
    const Scalar& p,
    int64_t dim,
    const Scalar& maxnorm) {
  Tensor out;
  renorm_meta(self, p, dim, maxnorm, out);
  renorm_impl(self, p, dim, maxnorm, out);
  return out;
}
} // namespace at
