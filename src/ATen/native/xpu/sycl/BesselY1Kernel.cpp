#include <ATen/Dispatch.h>
#include <ATen/native/xpu/sycl/MathExtensions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/core/Scalar.h>

#include <ATen/native/xpu/sycl/BesselY1Kernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct BesselY1Functor {
  scalar_t operator()(scalar_t a) const {
    return bessel_y1_forward(a);
  }
};

void bessel_y1_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "bessel_y1_xpu", [&]() {
    gpu_kernel(iter, BesselY1Functor<scalar_t>());
  });
}

}
