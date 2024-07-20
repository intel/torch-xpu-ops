#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AtanComplexFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(const scalar_t a) const {
    return std::atan(static_cast<opmath_t>(a));
  }
};

template <typename scalar_t>
struct AtanFunctor {
  scalar_t operator()(const scalar_t a) const {
    return std::atan(a);
  }
};

void atan_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "atan_xpu", [&]() {
          gpu_kernel(iter, AtanComplexFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "atan_xpu",
        [&]() { gpu_kernel(iter, AtanFunctor<scalar_t>()); });
  }
}

} // namespace at::native::xpu
