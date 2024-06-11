#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>
#include <comm/XPUMathCompat.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct SiluFunctor {
  scalar_t operator()(scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t x_acc = static_cast<opmath_t>(x);
          return x_acc / (opmath_t(1) + ::exp(-x_acc));
  }
};

void silu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "silu_xpu",
      [&]() {
        gpu_kernel(iter, SiluFunctor<scalar_t>()); });
}

} // namespace xpu
} // namespace native
} // namespace at
