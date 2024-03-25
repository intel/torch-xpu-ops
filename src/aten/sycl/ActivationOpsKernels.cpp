#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>
#include <comm/XPUMathCompat.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct ReluFunctor {
  scalar_t operator()(scalar_t x) const {
    return x <= scalar_t{0} ? scalar_t{0} : x;
  }
};

void relu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "relu_xpu", [&]() {
    gpu_kernel(iter, ReluFunctor<scalar_t>());
  });
}

} // namespace xpu
} // namespace native
} // namespace at
