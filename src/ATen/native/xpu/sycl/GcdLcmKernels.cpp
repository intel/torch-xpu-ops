#include <ATen/Dispatch.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct GcdFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return calc_gcd(a, b);
  }
};

void gcd_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_xpu", [&]() {
    gpu_kernel(iter, GcdFunctor<scalar_t>());
  });
}

} // namespace at::native::xpu