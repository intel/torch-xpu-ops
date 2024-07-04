#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AsinhFunctor {
  scalar_t operator()(const scalar_t a) const {
    using opmath_t = at::opmath_type<scalar_t>;
    return std::asinh(static_cast<opmath_t>(a));
  }
};

void asinh_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "asinh_xpu", [&]() {
          gpu_kernel(iter, AsinhFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "asinh_xpu",
        [&]() { gpu_kernel(iter, AsinhFunctor<scalar_t>()); });
  }
}

} // namespace at::native::xpu
