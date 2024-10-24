#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/xpu/sycl/LaguerrePolynomialLKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct laguerre_polynomial_l_functor {
  scalar_t operator()(scalar_t x, scalar_t n_) const {
    int64_t n = static_cast<int64_t>(n_);
    if (n < 0) {
      return scalar_t(0.0);
    }

    if (std::abs(x) == scalar_t(0.0)) {
      return scalar_t(1.0);
    }

    if (n == 0) {
      return scalar_t(1.0);
    }

    if (n == 1) {
      return scalar_t(1.0) - x;
    }

    scalar_t p = scalar_t(1.0);
    scalar_t q = scalar_t(1.0) - x;
    scalar_t r;

    for (int64_t k = 1; k < n; k++) {
      r = (((k + k) + (scalar_t(1.0) - x)) * q - k * p) / (k + 1);
      p = q;
      q = r;
    }

    return r;
  }
};

void laguerre_polynomial_l_kernel(TensorIteratorBase& iterator) {
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "laguerre_polynomial_l_xpu", [&]() {
        gpu_kernel_with_scalars(
            iterator, laguerre_polynomial_l_functor<scalar_t>());
      });
}

} // namespace at::native::xpu
