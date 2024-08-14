#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct PreluFunctor {
  scalar_t operator()(scalar_t input, scalar_t weight) const {
    return (input > 0) ? input : weight * input;
  }
};

template <typename scalar_t>
struct PreluBackwardFunctor {
  std::tuple<scalar_t, scalar_t> operator()(
      scalar_t input,
      scalar_t weight,
      scalar_t grad) const {
    auto mask = input > 0;
    auto grad_input = mask ? grad : weight * grad;
    auto grad_weight = mask ? scalar_t{0} : input * grad;
    return std::tuple<scalar_t, scalar_t>{grad_input, grad_weight};
  }
};

void prelu_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "prelu_xpu", [&] {
        gpu_kernel(iter, PreluFunctor<scalar_t>());
      });
}

void prelu_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "prelu_backward_xpu", [&] {
        gpu_kernel_multiple_outputs(iter, PreluBackwardFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu