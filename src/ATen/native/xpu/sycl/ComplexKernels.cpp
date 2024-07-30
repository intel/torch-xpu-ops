#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct ComplexFunctor {
  c10::complex<scalar_t> operator()(scalar_t a, scalar_t b) const {
    return c10::complex<scalar_t>(a, b);
  }
};

void complex_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      kHalf, iter.input_dtype(0), "complex_xpu", [&]() {
        ComplexFunctor<scalar_t> f;
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t>
struct PolarFunctor {
  c10::complex<scalar_t> operator()(scalar_t a, scalar_t b) const {
    return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
  }
};

void polar_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(0), "polar_xpu", [&]() {
    PolarFunctor<scalar_t> f;
    gpu_kernel(iter, f);
  });
}

} // namespace at::native::xpu
