
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/IGammaKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MathExtensions.h>

namespace at::native::xpu {

template <typename scalar_t>
struct IgammaFunctor {
  IgammaFunctor(bool calc_igammac) : calc_igammac_(calc_igammac) {}
  bool calc_igammac_;
  [[clang::optnone]] scalar_t operator()(scalar_t a, scalar_t b) const {
    if (calc_igammac_) {
      return calc_igammac<scalar_t>(a, b);
    } else {
      return calc_igamma<scalar_t>(a, b);
    }
  }
};

void igamma_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igamma_xpu", [&]() {
    gpu_kernel(iter, IgammaFunctor<scalar_t>(false));
  });
}

void igammac_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "igammac_xpu", [&]() {
    gpu_kernel(iter, IgammaFunctor<scalar_t>(true));
  });
}

} // namespace at::native::xpu
