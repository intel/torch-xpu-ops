
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/IGammaKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MathExtensions.h>

namespace at::native::xpu {

template <typename scalar_t>
struct IgammaFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return calc_igamma<scalar_t>(a, b);
  }
};

template <typename scalar_t>
struct IgammacFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return calc_igammac<scalar_t>(a, b);
  }
};

void igamma_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igamma_xpu", [&]() {
    gpu_kernel(iter, IgammaFunctor<scalar_t>());
  });
}

void igammac_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_xpu", [&]() {
    gpu_kernel(iter, IgammacFunctor<scalar_t>());
  });
}

} // namespace at::native::xpu
