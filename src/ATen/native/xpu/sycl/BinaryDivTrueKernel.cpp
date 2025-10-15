#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/BinaryInternal.h>
#include <ATen/native/xpu/sycl/BinaryKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

void div_true_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (iter.common_dtype() == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(iter, DivFunctor<opmath_t>());
    return;
  }
  if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_V2(
        common_dtype,
        "div_true_xpu",
        AT_WRAP([&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          auto inv_b = opmath_t(1.0) / iter.scalar_value<opmath_t>(2);
          iter.remove_operand(2);
          gpu_kernel(
              iter,
              BUnaryFunctor<scalar_t, scalar_t, scalar_t, MulFunctor<opmath_t>>(
                  MulFunctor<opmath_t>(), inv_b));
        }),
        AT_EXPAND(AT_COMPLEX_TYPES),
        AT_EXPAND(AT_FLOATING_TYPES),
        kHalf,
        kBFloat16,
        kFloat8_e5m2,
        kFloat8_e4m3fn,
        kFloat8_e5m2fnuz,
        kFloat8_e4m3fnuz);
  } else {
    AT_DISPATCH_V2(
        common_dtype,
        "div_true_xpu",
        AT_WRAP([&]() {
          DivFunctor<scalar_t> f;
          gpu_kernel_with_scalars(iter, f);
        }),
        AT_EXPAND(AT_COMPLEX_TYPES),
        AT_EXPAND(AT_FLOATING_TYPES),
        kHalf,
        kBFloat16,
        kFloat8_e5m2,
        kFloat8_e4m3fn,
        kFloat8_e5m2fnuz,
        kFloat8_e4m3fnuz);
  }
}

} // namespace at::native::xpu
