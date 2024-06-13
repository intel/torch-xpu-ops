#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
<<<<<<< HEAD:src/aten/sycl/ReduceLogicKernel.cpp
#include <aten/sycl/Reduce.h>
=======
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>
>>>>>>> 9222b7f (Align file structure to PyTorch):src/ATen/native/xpu/sycl/ReduceLogicKernel.cpp

namespace at::native::xpu {

template <typename scalar_t>
struct AndFunctor {
  inline bool operator()(scalar_t a, scalar_t b) const {
    return (static_cast<bool>(a) && static_cast<bool>(b));
  }
};

void and_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "and_xpu", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter, func_wrapper<bool>(AndFunctor<scalar_t>()), true);
      });
}

template <typename scalar_t>
struct OrFunctor {
  inline bool operator()(scalar_t a, scalar_t b) const {
    return (static_cast<bool>(a) || static_cast<bool>(b));
  }
};

void or_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "or_xpu", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter, func_wrapper<bool>(OrFunctor<scalar_t>()), false);
      });
}

} // namespace at::native::xpu
