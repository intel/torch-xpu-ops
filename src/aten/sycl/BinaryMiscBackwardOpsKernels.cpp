#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct TanhBackwardComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    using comp_t = at::opmath_type<scalar_t>;
    const auto one = comp_t{1.};
    const auto comp_b = static_cast<comp_t>(b);
    const auto comp_a = static_cast<comp_t>(a);
    return static_cast<scalar_t>(comp_a * std::conj(one - comp_b * comp_b));
  }
};

template <typename scalar_t>
struct TanhBackwardFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * (scalar_t{1.} - b * b);
  }
};

void tanh_backward_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, dtype, "tanh_backward_complex_xpu", [&]() {
          gpu_kernel(iter, TanhBackwardComplexFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "tanh_backward_xpu",
        [&]() { gpu_kernel(iter, TanhBackwardFunctor<scalar_t>()); });
  }
}

} // namespace xpu
} // namespace native
} // namespace at
