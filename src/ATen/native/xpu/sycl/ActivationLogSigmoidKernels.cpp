#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LogSigmoidForwardFunctor {
  bool operator()(scalar_t in_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t in = in_;
    const auto min = std::min(opmath_t(0), in);
    const auto z = std::exp(-std::abs(in));
    return min - std::log1p(z);
  }
};

void log_sigmoid_forward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "log_sigmoid_forward_xpu",
      [&]() { gpu_kernel(iter, LogSigmoidForwardFunctor<scalar_t>()); });
}

template <typename scalar_t>
struct LogSigmoidBackwardFunctor {
  bool operator()(scalar_t in_, scalar_t grad_out_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t in = in_;
    const opmath_t grad_out = grad_out_;

    bool in_negative = in < opmath_t(0);
    opmath_t max_deriv = in_negative ? opmath_t(1) : opmath_t(0);
    opmath_t sign = in_negative ? opmath_t(1) : -opmath_t(1);
    const opmath_t z = std::exp(-std::abs(in));
    return grad_out * (max_deriv - sign * (z / (opmath_t(1) + z)));
  }
};

void log_sigmoid_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "log_sigmoid_backward_xpu",
      [&]() { gpu_kernel(iter, LogSigmoidBackwardFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
