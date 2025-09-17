#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/xpu/sycl/HermitePolynomialHeKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct HermitePolynomialHeFunctor {
  scalar_t operator()(scalar_t x, scalar_t n_) const {
    int64_t n = static_cast<int64_t>(n_);
    if (n < 0) {
      return scalar_t(0.0);
    }

    if (n == 0) {
      return scalar_t(1.0);
    }

    if (n == 1) {
      return x;
    }

    scalar_t p = scalar_t(1.0);
    scalar_t q = x;
    scalar_t r;

    for (int64_t k = 1; (k < n) && !std::isnan(q); k++) {
      r = x * q - k * p;
      p = q;
      q = r;
    }

    return r;
  }
};

void hermite_polynomial_he_kernel(TensorIteratorBase& iterator) {
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "hermite_polynomial_he_xpu", [&]() {
        gpu_kernel_with_scalars(
            iterator, HermitePolynomialHeFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu
