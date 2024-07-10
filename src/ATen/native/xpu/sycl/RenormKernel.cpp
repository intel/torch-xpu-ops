#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/xpu/sycl/Loops.h>
namespace at::native::xpu {

template <typename scalar_t>
struct RenormOutFunctor {
  scalar_t operator()(scalar_t norm) const {
    const auto eps = static_cast<scalar_t>(1e-7);
    const auto one = static_cast<scalar_t>(1.0);
    return (norm > maxnorm_elm) ? maxnorm_elm / (norm + eps) : one;
  }

  RenormOutFunctor(scalar_t maxnorm_elm) : maxnorm_elm(maxnorm_elm) {}

 private:
  scalar_t maxnorm_elm;
};

Tensor& renorm_kernel(
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
  auto acc_type = at::toAccumulateType(dtype, /*is_cuda=*/true);
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

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "renorm_xpu",
      [&] {
        auto maxnorm_elm = maxnorm.to<scalar_t>();
        RenormOutFunctor<scalar_t> f(maxnorm_elm);
        gpu_kernel(iter, f);
      });
  return at::mul_outf(self, factor, const_cast<Tensor&>(out));
}
} // namespace at::native::xpu