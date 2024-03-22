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

} // namespace xpu
} // namespace native
} // namespace at
