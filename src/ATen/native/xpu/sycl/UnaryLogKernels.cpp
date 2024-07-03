#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/CopyKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LogFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::log(a);
  }
};

void log_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, iter.common_dtype(), "log_xpu", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel(iter, LogFunctor<opmath_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        iter.common_dtype(),
        "log_xpu",
        [&]() { gpu_kernel(iter, LogFunctor<scalar_t>()); });
  }
}

} // namespace at::native::xpu
