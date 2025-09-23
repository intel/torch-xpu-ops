#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/XPUMathCompat.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ActivationHardswishKernels.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct HardswishFunctor {
  scalar_t operator()(scalar_t self_val) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t zero(0.0f);
    const opmath_t one_sixth(1.0f / 6.0f);
    const opmath_t three(3.0f);
    const opmath_t six(6.0f);
    opmath_t x = static_cast<opmath_t>(self_val);
    return x * std::min(std::max(x + three, zero), six) * one_sixth;
  }
};

template <typename scalar_t>
struct HardswishBackwardFunctor {
  scalar_t operator()(scalar_t grad_val_, scalar_t self_val_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t zero(0.0f);
    const opmath_t three(3.0f);
    const opmath_t neg_three(-3.0f);
    const opmath_t one_half(0.5f);
    opmath_t grad_val = static_cast<opmath_t>(grad_val_);
    opmath_t self_val = static_cast<opmath_t>(self_val_);
    if (self_val <= neg_three) {
      return zero;
    } else if (self_val < three) {
      return grad_val * ((self_val / three) + one_half);
    } else {
      return grad_val;
    }
  }
};

void hardswish_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardswish_xpu",
      [&]() { gpu_kernel(iter, HardswishFunctor<scalar_t>()); });
}

void hardswish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardswish_backward_xpu",
      [&]() { gpu_kernel(iter, HardswishBackwardFunctor<scalar_t>()); });
}

} // namespace xpu
} // namespace native
} // namespace at
