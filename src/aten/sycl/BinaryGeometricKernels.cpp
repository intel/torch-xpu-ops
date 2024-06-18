#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct HypotFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::hypot(a, b);
  }
};

void hypot_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "hypot_xpu",
      [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
            iter, HypotFunctor<scalar_t>());
      });
}

} // namespace xpu
} // namespace native
} // namespace at
