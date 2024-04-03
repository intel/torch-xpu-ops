#include <ATen/Dispatch.h>
// #include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/Math.h>

namespace at::native::xpu {

template <typename scalar_t>
struct GcdFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return calc_gcd(a, b);
  }
};

void gcd_kernel_xpu(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_xpu", [&]() {
    gpu_kernel(iter, GcdFunctor<scalar_t>());
  });
}

} // namespace at::native::xpu