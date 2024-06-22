#include <ATen/ATen.h>
#include <ATen/OpMathType.h>

#include <aten/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AddcmulKernelFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return static_cast<opmath_t>(a) +
        alpha * static_cast<opmath_t>(b) * static_cast<opmath_t>(c);
  }

  AddcmulKernelFunctor(opmath_t alpha) : alpha(alpha) {}

 private:
  opmath_t alpha;
};

void addcmul_kernel(TensorIterator& iter, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "addcmul_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto alpha = value.to<opmath_t>();
        AddcmulKernelFunctor<scalar_t> f(alpha);
        gpu_kernel(iter, f);
      });
}

} // namespace at::native::xpu
