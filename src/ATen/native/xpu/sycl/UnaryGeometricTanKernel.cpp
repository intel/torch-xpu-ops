#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct TanComplexFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t a) const {
    return std::tan(static_cast<opmath_t>(a));
  }
};

template <typename scalar_t>
struct TanFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::tan(a);
  }
};

void tan_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "tan_xpu", [&]() {
      gpu_kernel(iter, TanComplexFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, common_dtype, "tan_xpu", [&]() {
          gpu_kernel(iter, TanFunctor<scalar_t>());
        });
  }
}

} // namespace at::native::xpu
