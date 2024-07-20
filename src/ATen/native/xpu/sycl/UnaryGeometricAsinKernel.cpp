#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AsinComplexFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(const scalar_t a) const {
    return std::asin(static_cast<opmath_t>(a));
  }
};

template <typename scalar_t>
struct AsinFunctor {
  scalar_t operator()(const scalar_t a) const {
    return std::asin(a);
  }
};

void asin_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "asin_xpu", [&]() {
          gpu_kernel(iter, AsinComplexFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "asin_xpu",
        [&]() { gpu_kernel(iter, AsinFunctor<scalar_t>()); });
  }
}

} // namespace at::native::xpu
