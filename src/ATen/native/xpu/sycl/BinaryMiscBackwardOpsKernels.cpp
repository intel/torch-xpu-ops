#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SigmoidBackwardComplexFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto one = opmath_t{1.};
    const auto comp_b = static_cast<opmath_t>(b);
    const auto comp_a = static_cast<opmath_t>(a);
    return static_cast<scalar_t>(comp_a * std::conj((one - comp_b) * comp_b));
  }
};

template <typename scalar_t>
struct SigmoidBackwardFloatFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * (scalar_t(1.) - b) * b;
  }
};

void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, dtype, "sigmoid_backward_xpu", [&]() {
          gpu_kernel(iter, SigmoidBackwardComplexFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "sigmoid_backward_xpu",
        [&]() { gpu_kernel(iter, SigmoidBackwardFloatFunctor<scalar_t>()); });
  }
}

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

} // namespace at::native::xpu
