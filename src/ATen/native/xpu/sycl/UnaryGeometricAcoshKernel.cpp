#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t, typename acc_t = scalar_t>
struct AcoshFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::acosh(static_cast<acc_t>(a));
  }
};

void acosh_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "acosh_xpu", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          auto caller = AcoshFunctor<scalar_t, opmath_t>();
          gpu_kernel(iter, caller);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        common_dtype,
        "acosh_xpu",
        [&]() {
          auto caller = AcoshFunctor<scalar_t>();
          gpu_kernel(iter, caller);
        });
  }
}

} // namespace at::native::xpu
