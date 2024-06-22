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
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "sigmoid_xpu",
      [&]() { gpu_kernel(iter, SigmoidFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct ErfFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::erf(float(a));
  }
};

template <>
struct ErfFunctor<double> {
  double operator()(double a) const {
    return std::erf(a);
  }
};

template <typename scalar_t>
struct ErfcFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::erfc(float(a));
  }
};

template <>
struct ErfcFunctor<double> {
  double operator()(double a) const {
    return std::erfc(a);
  }
};

void erf_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "erf_xpu",
      [&]() { gpu_kernel(iter, ErfFunctor<scalar_t>()); });
}

void erfc_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      iter.common_dtype(),
      "erfc_xpu",
      [&]() { gpu_kernel(iter, ErfcFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
