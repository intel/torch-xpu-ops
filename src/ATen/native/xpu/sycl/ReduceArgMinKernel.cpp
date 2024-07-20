#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <ATen/native/xpu/sycl/Reduce.h>

namespace at::native::xpu {

template <typename scalar_t, typename acc_t = scalar_t>
void argmin_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
      iter,
      ArgMinOps<acc_t>{},
      std::pair<acc_t, int64_t>(at::numeric_limits<acc_t>::upper_bound(), 0));
};

void argmin_kernel(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    argmin_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kBFloat16) {
    argmin_kernel_impl<at::BFloat16, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmin_xpu", [&]() {
      argmin_kernel_impl<scalar_t>(iter);
    });
  }
}

} // namespace at::native::xpu
