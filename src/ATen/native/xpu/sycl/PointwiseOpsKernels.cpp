#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AddcmulKernelFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return static_cast<opmath_t>(a) +
        alpha_ * static_cast<opmath_t>(b) * static_cast<opmath_t>(c);
  }

  AddcmulKernelFunctor(opmath_t alpha) : alpha_(alpha) {}

 private:
  opmath_t alpha_;
};

void addcmul_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "addcmul_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto alpha = value.to<opmath_t>();
        AddcmulKernelFunctor<scalar_t> f(alpha);
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t>
struct MSEBackwardFunctor {
  scalar_t operator()(scalar_t a, scalar_t b, scalar_t c) const {
    return alpha_ * (a - b) * c;
  }
  MSEBackwardFunctor(scalar_t alpha) : alpha_(alpha) {}

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
        gpu_kernel(iter, MSEBackwardFunctor<scalar_t>(alpha));
      });
}

} // namespace at::native::xpu
