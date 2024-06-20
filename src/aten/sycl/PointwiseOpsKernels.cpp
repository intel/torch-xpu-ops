#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct MSEBackwardKernelFunctor {
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return alpha_ * (a - b) * c;
  }
  MSEBackwardKernelFunctor(scalar_t alpha) : alpha_(alpha) {}

 private:
  scalar_t alpha_;
};

void mse_backward_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mse_backward_xpu",
      [&]() {
        auto alpha = value.to<scalar_t>();
        gpu_kernel(iter, MSEBackwardKernelFunctor<scalar_t>(alpha));
      });
}

} // namespace at::native::xpu
