#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/ActivationSoftshrinkKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SoftshrinkFunctor {
  scalar_t operator()(scalar_t a) const {
    return at::_isnan(a)
        ? a
        : (a > lambd_ ? a - lambd_ : (a < -lambd_ ? a + lambd_ : scalar_t(0)));
  }

  SoftshrinkFunctor(scalar_t lambd) : lambd_(lambd) {}

 private:
  scalar_t lambd_;
};

void softshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "softshrink_xpu",
      [&]() {
        auto lambd = value.to<scalar_t>();
        SoftshrinkFunctor<scalar_t> f(lambd);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t>
struct SoftshrinkBackwardFunctor {
  scalar_t operator()(scalar_t grad_val, scalar_t self_val) const {
    return (self_val >= -lambd_ && self_val <= lambd_) ? scalar_t(0) : grad_val;
  }

  SoftshrinkBackwardFunctor(scalar_t lambd) : lambd_(lambd) {}

 private:
  scalar_t lambd_;
};

void softshrink_backward_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "shrink_backward_xpu",
      [&]() {
        auto lambd = value.to<scalar_t>();
        SoftshrinkBackwardFunctor<scalar_t> f(lambd);
        gpu_kernel(iter, f);
      });
}

} // namespace at::native::xpu
