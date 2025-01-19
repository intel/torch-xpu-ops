#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/SharedReduceOps.h>

#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>

#include <ATen/native/xpu/sycl/ReduceMinValuesKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename acc_t>
struct MinNanFunctor {
  inline acc_t operator()(acc_t a, acc_t b) const {
    return (at::_isnan(a) || a < b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
void min_values_kernel_xpu_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      func_wrapper<acc_t>(MinNanFunctor<acc_t>()),
      at::numeric_limits<acc_t>::upper_bound());
}

void min_values_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(), "min_values_xpu", [&]() {
        min_values_kernel_xpu_impl<scalar_t>(iter);
      });
}

void min_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "min_xpu", [&]() {
        gpu_reduce_kernel<scalar_t, scalar_t, 4, 2>(
            iter,
            MinOps<scalar_t>{},
            at::xpu::pair<scalar_t, int64_t>(
                at::numeric_limits<scalar_t>::upper_bound(), 0));
      });
}

void min_all_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "min_all_xpu", [&] {
        min_values_kernel_xpu_impl<scalar_t>(iter);
      });
}

} // namespace xpu
} // namespace native
} // namespace at
