#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/core/Scalar.h>

#include <ATen/native/xpu/sycl/ZetaKernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct ZetaFunctor {
  scalar_t operator()(scalar_t x, scalar_t q) const {
    return zeta<scalar_t, /*is_xpu=*/true>(x, q);
  }
};

constexpr char zeta_name[] = "zeta";
void zeta_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "zeta_xpu", [&]() {
    gpu_kernel_with_scalars(iter, ZetaFunctor<scalar_t>());
  });
}

} // namespace at::native::xpu
