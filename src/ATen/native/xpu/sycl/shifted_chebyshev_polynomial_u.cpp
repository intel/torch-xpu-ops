#include <ATen/ATen.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/shifted_chebyshev_polynomial_u.h>

namespace at::native::xpu {

template <typename scalar_t>
struct shifted_chebyshev_polynomial_u_functor {
  scalar_t operator()(scalar_t x, scalar_t n) const {
    return shifted_chebyshev_polynomial_u_forward<scalar_t>(x, n);
  }
};

void shifted_chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator) {
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "shifted_chebyshev_polynomial_u_xpu", [&]() {
        shifted_chebyshev_polynomial_u_functor<scalar_t> f;
        gpu_kernel_with_scalars(iterator, f);
      });
}

} // namespace at::native::xpu