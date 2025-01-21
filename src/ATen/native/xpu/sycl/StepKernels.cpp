#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/BFloat16-math.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/StepKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct NextafterFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::nextafter(a, b);
  }
};

template <typename scalar_t>
struct HeavisideFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a == 0 ? b : static_cast<scalar_t>(a > 0);
  }
};

void nextafter_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "nextafter_xpu", [&]() {
        gpu_kernel_with_scalars(iter, NextafterFunctor<scalar_t>());
      });
}

void heaviside_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_xpu", [&]() {
        gpu_kernel_with_scalars(iter, HeavisideFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu
