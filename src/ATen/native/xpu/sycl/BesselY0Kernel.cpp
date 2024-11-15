#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/core/Scalar.h>

#include <ATen/native/xpu/sycl/BesselY0Kernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct BesselY0Functor {
  scalar_t operator()(scalar_t a) const {
    return bessel_y0_forward(a);
  }
};

void bessel_y0_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "bessel_y0_xpu", [&]() {
    gpu_kernel(iter, BesselY0Functor<scalar_t>());
  });
}

} // namespace at::native::xpu
