#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct CopysignFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::copysign(a, b);
  }
};

void copysign_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "copysign_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel_with_scalars(iter, CopysignFunctor<opmath_t>());
      });
}

} // namespace at::native::xpu
