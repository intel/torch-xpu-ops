#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/core/Scalar.h>

#include <ATen/native/xpu/sycl/SphericalBesselJ0Kernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SphericalBesselJ0Functor {
  scalar_t operator()(scalar_t a) const {
    return spherical_bessel_j0_forward(a);
  }
};

void spherical_bessel_j0_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(
      iter.common_dtype(), "spherical_bessel_j0_xpu", [&]() {
        gpu_kernel(iter, SphericalBesselJ0Functor<scalar_t>());
      });
}

} // namespace at::native::xpu
