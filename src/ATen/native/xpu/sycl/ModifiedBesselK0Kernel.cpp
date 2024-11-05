#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/core/Scalar.h>

#include <ATen/native/xpu/sycl/ModifiedBesselK0Kernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct ModifiedBesselK0Functor {
  scalar_t operator()(scalar_t a) const {
    return modified_bessel_k0_forward(a);
  }
};

void modified_bessel_k0_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(
      iter.common_dtype(), "modified_bessel_k0_xpu", [&]() {
        gpu_kernel(iter, ModifiedBesselK0Functor<scalar_t>());
      });
}

} // namespace at::native::xpu
