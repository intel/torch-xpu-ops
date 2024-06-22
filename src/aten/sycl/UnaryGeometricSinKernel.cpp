#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <aten/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SinFunctor {
  scalar_t operator()(const scalar_t a) const {
    return std::sin(a);
  }
};

void sin_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "sin_xpu", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(iter, SinFunctor<opmath_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, common_dtype, "sin_xpu", [&]() {
          gpu_kernel(iter, SinFunctor<scalar_t>());
        });
  }
}

} // namespace at::native::xpu
