#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <aten/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SigmoidFunctor {
  scalar_t operator()(scalar_t a) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto one = opmath_t{1.0};
    return one / (one + std::exp(-static_cast<opmath_t>(a)));
  }
};

void sigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      kComplexHalf,
      iter.common_dtype(),
      "sigmoid_xpu",
      [&]() { gpu_kernel(iter, SigmoidFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
