#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/xpu/sycl/Loops.h>
namespace at::native::xpu {

template <typename scalar_t>
struct RenormScalarFactorFunctor {
  scalar_t operator()(scalar_t norm) const {
    const auto eps = static_cast<scalar_t>(1e-7);
    const auto one = static_cast<scalar_t>(1.0);
    return (norm > maxnorm_elm) ? maxnorm_elm / (norm + eps) : one;
  }

  RenormScalarFactorFunctor(scalar_t maxnorm_elm) : maxnorm_elm(maxnorm_elm) {}

 private:
  scalar_t maxnorm_elm;
};

Tensor& renorm_scale_factor_kernel(TensorIteratorBase& iter, double maxnorm) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "renorm_scale_factor_xpu",
      [&] {
        auto maxnorm_elm = maxnorm.to<scalar_t>();
        RenormScalarFactorFunctor<scalar_t> f(maxnorm_elm);
        gpu_kernel(iter, f);
      });
}

} // namespace at::native::xpu
