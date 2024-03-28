#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/ReduceOps.h>
#include <aten/sycl/NumericLimits.h>
#include <aten/sycl/Reduce.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, typename acc_t = scalar_t>
void argmax_kernel_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
      iter,
      ArgMaxOps<acc_t>{},
      std::pair<acc_t, int64_t>(at::numeric_limits<acc_t>::lower_bound(), 0));
};

void argmax_kernel(TensorIterator& iter) {
  if (iter.dtype(1) == kHalf) {
    argmax_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kBFloat16) {
    argmax_kernel_impl<at::BFloat16, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmax_xpu", [&]() {
      argmax_kernel_impl<scalar_t>(iter);
    });
  }
}

} // namespace xpu
} // namespace native
} // namespace at
